// Ported verbatim from async_mpc/fields/src/traits/sqrt.rs.
// Square root over the Mersenne-61 tower (Fp, Fp2, Fp4) via Scott's complex
// method. Needed because lambdaworks's stock `FieldElement::sqrt` is gated on
// `IsPrimeField` and so doesn't apply to the Fp2 / Fp4 extension types — but
// rand_bit needs sqrt over the protocol's `LargeField` (= Fp4_61).

use lambdaworks_math::field::element::FieldElement;

use super::{
    extensions::{Mersenne61Degree2ExtensionField, Mersenne61Degree4ExtensionField},
    field::Mersenne61Field,
};

type FpE = FieldElement<Mersenne61Field>;
type Fp2E = FieldElement<Mersenne61Degree2ExtensionField>;
type Fp4E = FieldElement<Mersenne61Degree4ExtensionField>;

/// Square root over the Mersenne-61 tower (Fp, Fp2, Fp4).
///
/// For input `a`, returns `Some((r, -r))` such that `r * r == a`, or `None`
/// if `a` is a quadratic non-residue. `sqrt(0) == (0, 0)`.
///
/// Implementation uses Scott's "complex method" (Scott 2007) recursively:
/// Fp4 sqrt reduces to two Fp2 sqrts (norm + half-discriminant); each Fp2
/// sqrt reduces to two Fp sqrts. The base case (`FieldElement<Mersenne61Field>`)
/// uses lambdaworks's inherent `sqrt` (Tonelli–Shanks) via method resolution.
pub trait Sqrt: Sized {
    fn sqrt(&self) -> Option<(Self, Self)>;
}

// ---------------------------------------------------------------------------
// Fp2 = Fp[i] / (i^2 + 1)
//
// Solve s² = c for s = s0 + s1·i given c = c0 + c1·i:
//   s0² − s1² = c0
//   2·s0·s1   = c1
// Norm N(c) = c0² + c1² must be a square in Fp; let t = sqrt_Fp(N(c)).
// Then s0² ∈ {(c0 + t)/2, (c0 − t)/2}, where exactly one is a square in Fp
// (the other equals −s1², a non-residue since −1 is non-residue in Fp).
// ---------------------------------------------------------------------------
impl Sqrt for Fp2E {
    fn sqrt(&self) -> Option<(Self, Self)> {
        let [c0, c1] = self.value().clone();

        if c0 == FpE::zero() && c1 == FpE::zero() {
            return Some((Fp2E::zero(), Fp2E::zero()));
        }

        if c1 == FpE::zero() {
            if let Some((r, _)) = c0.sqrt() {
                let s = Fp2E::new([r.clone(), FpE::zero()]);
                return Some((s.clone(), -s));
            }
            // c0 non-residue in Fp ⇒ −c0 is a residue; s = r·i with r² = −c0.
            let (r, _) = (-c0).sqrt()?;
            let s = Fp2E::new([FpE::zero(), r]);
            return Some((s.clone(), -s));
        }

        let norm = c0.square() + c1.square();
        let (t, _) = norm.sqrt()?;
        let two_inv = FpE::from(2u64).inv().unwrap();

        let s0 = match ((c0.clone() + &t) * &two_inv).sqrt() {
            Some((r, _)) => r,
            None => ((c0 - &t) * &two_inv).sqrt()?.0,
        };
        let s1 = c1 * (FpE::from(2u64) * &s0).inv().unwrap();
        let s = Fp2E::new([s0, s1]);
        Some((s.clone(), -s))
    }
}

// ---------------------------------------------------------------------------
// Fp4 = Fp2[w] / (w^2 − (4 + i))
//
// Same recursion with δ = 4 + i ∈ Fp2:
//   s0² + δ·s1² = c0
//   2·s0·s1     = c1
// Norm N(c) = c0² − δ·c1² must be a square in Fp2. One of
// {(c0 + t)/2, (c0 − t)/2} equals s0² (a square), the other equals δ·s1²
// (non-residue: square · non-residue, since δ is a non-residue in Fp2).
// ---------------------------------------------------------------------------
impl Sqrt for Fp4E {
    fn sqrt(&self) -> Option<(Self, Self)> {
        let [c0, c1] = self.value().clone();
        let delta = Fp2E::new([FpE::from(4u64), FpE::one()]); // 4 + i

        if c0 == Fp2E::zero() && c1 == Fp2E::zero() {
            return Some((Fp4E::zero(), Fp4E::zero()));
        }

        if c1 == Fp2E::zero() {
            if let Some((r, _)) = c0.sqrt() {
                let s = Fp4E::new([r.clone(), Fp2E::zero()]);
                return Some((s.clone(), -s));
            }
            // c0 non-residue in Fp2 ⇒ c0/δ is a residue (δ also non-residue).
            let q = (c0 / &delta).ok()?;
            let (r, _) = q.sqrt()?;
            let s = Fp4E::new([Fp2E::zero(), r]);
            return Some((s.clone(), -s));
        }

        let norm = c0.square() - &delta * c1.square();
        let (t, _) = norm.sqrt()?;
        let two_inv = Fp2E::from(2u64).inv().unwrap();

        let s0 = match ((c0.clone() + &t) * &two_inv).sqrt() {
            Some((r, _)) => r,
            None => ((c0 - &t) * &two_inv).sqrt()?.0,
        };
        let s1 = c1 * (Fp2E::from(2u64) * &s0).inv().unwrap();
        let s = Fp4E::new([s0, s1]);
        Some((s.clone(), -s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn rand_fp(rng: &mut ChaCha20Rng) -> FpE {
        FpE::from(rng.next_u64())
    }
    fn rand_fp2(rng: &mut ChaCha20Rng) -> Fp2E {
        Fp2E::new([rand_fp(rng), rand_fp(rng)])
    }
    fn rand_fp4(rng: &mut ChaCha20Rng) -> Fp4E {
        Fp4E::new([rand_fp2(rng), rand_fp2(rng)])
    }

    #[test]
    fn fp2_sqrt_of_zero_is_zero_pair() {
        let (r0, r1) = Fp2E::zero().sqrt().unwrap();
        assert_eq!(r0, Fp2E::zero());
        assert_eq!(r1, Fp2E::zero());
    }

    #[test]
    fn fp2_sqrt_of_one() {
        let (r, neg_r) = Fp2E::one().sqrt().unwrap();
        assert_eq!(r.square(), Fp2E::one());
        assert_eq!(neg_r, -r);
    }

    #[test]
    fn fp2_known_nonresidue_returns_none() {
        // 4 + i is a proven non-residue in Fp2.
        let alpha = Fp2E::new([FpE::from(4u64), FpE::one()]);
        assert!(alpha.sqrt().is_none());
    }

    #[test]
    fn fp2_roundtrip_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF2D00D);
        for _ in 0..100 {
            let a = rand_fp2(&mut rng);
            let a_sq = a.square();
            let (r, neg_r) = a_sq.sqrt().expect("squared values are QR");
            assert_eq!(r.square(), a_sq);
            assert_eq!(neg_r, -r.clone());
            assert!(r == a || r == -&a, "root must equal ±a");
        }
    }

    #[test]
    fn fp4_sqrt_of_zero_is_zero_pair() {
        let (r0, r1) = Fp4E::zero().sqrt().unwrap();
        assert_eq!(r0, Fp4E::zero());
        assert_eq!(r1, Fp4E::zero());
    }

    #[test]
    fn fp4_sqrt_of_one() {
        let (r, neg_r) = Fp4E::one().sqrt().unwrap();
        assert_eq!(r.square(), Fp4E::one());
        assert_eq!(neg_r, -r);
    }

    #[test]
    fn fp4_sqrt_of_four_plus_i_is_plus_minus_w() {
        // w² = 4 + i by construction of the tower.
        let w = Fp4E::new([Fp2E::zero(), Fp2E::one()]);
        let four_plus_i = Fp4E::new([
            Fp2E::new([FpE::from(4u64), FpE::one()]),
            Fp2E::zero(),
        ]);
        let (r, neg_r) = four_plus_i.sqrt().unwrap();
        assert_eq!(r.square(), four_plus_i);
        assert_eq!(neg_r, -r.clone());
        assert!(r == w || r == -&w);
    }

    #[test]
    fn fp4_w_is_nonresidue() {
        let w = Fp4E::new([Fp2E::zero(), Fp2E::one()]);
        assert!(w.sqrt().is_none());
    }

    #[test]
    fn fp4_roundtrip_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF4D00D);
        for _ in 0..100 {
            let a = rand_fp4(&mut rng);
            let a_sq = a.square();
            let (r, neg_r) = a_sq.sqrt().expect("squared values are QR");
            assert_eq!(r.square(), a_sq);
            assert_eq!(neg_r, -r.clone());
            assert!(r == a || r == -&a, "root must equal ±a");
        }
    }
}
