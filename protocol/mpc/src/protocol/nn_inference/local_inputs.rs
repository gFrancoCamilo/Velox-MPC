//! Deriving this party's input shares locally, with no communication.
//!
//! For throughput extrapolation we want to time the online and output phases
//! without paying for the input phase, which at AlexNet scale is ~99% of all
//! sharings here (Velox does not pack, so every weight is its own sharing).
//! Each party instead computes its own share of every weight and activation.
//!
//! # Matching the ACSS encoding
//!
//! `acss_ab` builds each sharing as a degree-`t` polynomial through
//!
//! ```text
//!   x = [0, 1, 2, ..., t]
//!   y = [secret, prf_0, prf_1, ..., prf_{t-1}]
//! ```
//!
//! (`sample_polynomials_from_prf` puts the secret at 0 and one PRF sample per
//! party key at 1..=t), and party `i` receives the value at `i + 1`. Matching
//! that degree is what keeps the shares usable by the DN multiplication and the
//! output reconstruction downstream.
//!
//! Because `x` is the same for every sharing, a share is a fixed linear
//! combination of `y`, so it costs one dot product of length `t + 1`.
//!
//! # What this gives up
//!
//! The PRF samples are keyed per dealer-party pair, which is what makes the
//! sharing hiding. Here they are a public function of the value's global index
//! so every party derives the same polynomial without communicating. The shares
//! are correct and mutually consistent but **not secret** — benchmark only.

use protocol::LargeField;
use rayon::prelude::*;

/// Public stand-in for the per-dealer PRF samples, offset well clear of the
/// deterministic weight/activation values so a mis-indexed mask cannot look
/// like a plausible secret.
const MASK_BASE: u64 = 1 << 50;

fn mask(index: usize, slot: usize) -> LargeField {
    LargeField::from(MASK_BASE + (index as u64).wrapping_mul(1_000_003) + slot as u64)
}

/// Precomputed Lagrange coefficients for one party's evaluation point.
pub struct LocalShareDeriver {
    /// `t + 1` coefficients: `coeffs[0]` weights the secret, the rest the masks.
    coeffs: Vec<LargeField>,
    num_faults: usize,
}

impl LocalShareDeriver {
    pub fn new(my_id: usize, num_faults: usize) -> Self {
        // Interpolation points 0..=t, this party's share point my_id + 1.
        let xs: Vec<LargeField> = (0..=num_faults)
            .map(|i| LargeField::from(i as u64))
            .collect();
        let alpha = LargeField::from((my_id + 1) as u64);

        let coeffs = xs
            .iter()
            .enumerate()
            .map(|(k, xk)| {
                let mut num = LargeField::one();
                let mut den = LargeField::one();
                for (m, xm) in xs.iter().enumerate() {
                    if m == k {
                        continue;
                    }
                    num = num * (alpha.clone() - xm.clone());
                    den = den * (xk.clone() - xm.clone());
                }
                match den.inv() {
                    Ok(inv) => num * inv,
                    Err(_) => {
                        log::error!(
                            "LocalShareDeriver: duplicate interpolation point at {}; \
                             derived shares will be wrong",
                            k
                        );
                        LargeField::zero()
                    }
                }
            })
            .collect();

        log::info!(
            "LocalShareDeriver: party {} deriving shares locally (degree {} polynomials)",
            my_id, num_faults
        );
        Self { coeffs, num_faults }
    }

    /// This party's share of the sharing whose secret is `secret`, where
    /// `index` is the value's global index (it selects the masks).
    pub fn share_of(&self, index: usize, secret: LargeField) -> LargeField {
        let mut acc = self.coeffs[0].clone() * secret;
        for slot in 0..self.num_faults {
            acc = acc + self.coeffs[slot + 1].clone() * mask(index, slot);
        }
        acc
    }

    /// Shares for a contiguous run of global indices, in parallel.
    pub fn shares_range<F>(&self, start: usize, count: usize, secret_of: F) -> Vec<LargeField>
    where
        F: Fn(usize) -> LargeField + Sync + Send,
    {
        (0..count)
            .into_par_iter()
            .map(|i| self.share_of(start + i, secret_of(start + i)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n shares must lie on a degree-t polynomial whose value at 0 is the
    /// secret. Reconstructing from any t+1 of them must return it.
    #[test]
    fn shares_reconstruct_the_secret() {
        let (n, t) = (13usize, 4usize);
        let secret = LargeField::from(123_456_789u64);
        let idx = 77usize;

        let shares: Vec<LargeField> = (0..n)
            .map(|p| LocalShareDeriver::new(p, t).share_of(idx, secret.clone()))
            .collect();

        // Lagrange-interpolate the first t+1 shares back to x = 0.
        let xs: Vec<LargeField> = (1..=(t + 1)).map(|i| LargeField::from(i as u64)).collect();
        let zero = LargeField::from(0u64);
        let mut acc = LargeField::zero();
        for (k, xk) in xs.iter().enumerate() {
            let mut num = LargeField::one();
            let mut den = LargeField::one();
            for (m, xm) in xs.iter().enumerate() {
                if m == k { continue; }
                num = num * (zero.clone() - xm.clone());
                den = den * (xk.clone() - xm.clone());
            }
            acc = acc + shares[k].clone() * num * den.inv().unwrap();
        }
        assert_eq!(acc, secret, "reconstruction did not recover the secret");
    }

    /// Parties 0..t-1 sit on interpolation points 1..t, so their share is the
    /// mask itself — a direct check on the coefficient vector.
    #[test]
    fn low_parties_receive_their_mask_directly() {
        let t = 4usize;
        let secret = LargeField::from(42u64);
        for party in 0..t {
            let got = LocalShareDeriver::new(party, t).share_of(9, secret.clone());
            assert_eq!(got, mask(9, party), "party {} should receive mask {}", party, party);
        }
    }
}
