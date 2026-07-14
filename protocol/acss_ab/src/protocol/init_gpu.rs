//! GPU-accelerated ACSS dealer path (`--features gpu`).
//!
//! Mirrors async_mpc's `init_acss_gpu` pipeline, adapted to Velox's protocol
//! shape (all shares are 32-byte Fp4_61 `LargeField` elements; the DZK proof
//! is a broadcast coefficient vector instead of per-party evaluations):
//!
//! 1. PRF sampling on CPU (identical to the CPU dealer, nonce tags 1/2/3).
//! 2. Async H2D upload of the PRF evaluation matrix, overlapped with the
//!    nonce sampling and (first call only) the cached-matrix uploads.
//! 3. Two D2D GEMM chains — interpolation then evaluation — for the main
//!    batch (num_polys) and the nonce/blinding/blinding-nonce batch (3).
//! 4. On-device packing of per-party share rows (LE→BE wire conversion) and
//!    nonce append.
//! 5. Host-side AES commitments + Merkle roots (matching async_mpc's
//!    correctness fix: the fully-GPU Merkle path is kept dormant), then the
//!    Fiat-Shamir challenge.
//! 6. On-device DZK coefficient combination
//!    (`dzk[c] = blind[c] + Σ_j coeffs[j][c]·r^(j+1)`).
//! 7. Pinned D2H of the packed party rows; serialization + CTRBC/AVID via the
//!    shared `broadcast_dealer_artifacts` tail.
//!
//! Every step returns `Result`; on any CUDA failure `init_acss_ab_gpu` falls
//! back to the CPU dealer, so a GPU-built binary degrades gracefully.

use std::collections::HashMap;
use std::time::Instant;

use crypto::aes_hash::{HashState, MerkleTree};
use crypto::hash::Hash;
use protocol::gpu_acss_ffi::{append_nonce, compute_dzk_coeffs, pack_party_shares};
use protocol::gpu_mem::{device_to_host, CudaStream, DeviceBuffer, PinnedHostBuffer};
use protocol::ByteConversion;
use protocol::{
    inverse_vandermonde, powers_matrix, rand_field_element, sample_polynomials_from_prf,
    vandermonde_matrix, LargeField, LargeFieldSer, PreparedFieldGemm,
};
use rayon::prelude::*;
use types::Replica;

use super::init::{dealer_core_cpu, DealerArtifacts};
use crate::Context;

/// Serialized width of one `LargeField` (Fp4 over Mersenne-61).
const FIELD_BYTES: usize = 32;
/// Base-prime limb width for the per-limb LE↔BE wire conversion.
const LIMB_BYTES: usize = 8;

// ---------------------------------------------------------------------------
// Per-context GEMM cache
// ---------------------------------------------------------------------------

/// Reusable device state for the GPU dealer.
///
/// The two fixed matrices (inverse Vandermonde over points {0,1..t} and the
/// powers matrix over points 1..n) depend only on (t, n), which are constant
/// for the lifetime of a `Context`. Each matrix is uploaded into **two**
/// separate GEMM contexts — one for the main share batch and one for the
/// nonce batch — because a context's device output buffer is invalidated by
/// its next compute call, and the main interpolation output (`d_coeffs`) must
/// stay live until the DZK kernel runs.
pub struct AcssGemmCache {
    /// Interpolation GEMM for the share batch; its output is `d_coeffs`.
    inv_vand_main: Option<PreparedFieldGemm>,
    /// Evaluation GEMM for the share batch; its output is `d_evals`.
    powers_main: Option<PreparedFieldGemm>,
    /// Interpolation GEMM for the 3-row nonce batch; its output holds the
    /// blinding polynomial's coefficients (row 1) for the DZK kernel.
    inv_vand_nonce: Option<PreparedFieldGemm>,
    /// Evaluation GEMM for the 3-row nonce batch.
    powers_nonce: Option<PreparedFieldGemm>,
    /// (t, n) the cached matrices were built for.
    key: Option<(usize, usize)>,
    /// Device scratch: PRF evaluation matrix, num_polys × (t+1) × 32 B.
    d_y_matrix: DeviceBuffer,
    /// Device scratch: nonce PRF rows, 3 × (t+1) × 32 B.
    d_nonce_prfs: DeviceBuffer,
    /// Device scratch: packed per-party wire rows, n × party_stride.
    d_party_bytes: DeviceBuffer,
    /// Device scratch: DZK challenge powers, num_polys × 32 B.
    d_dzk_powers: DeviceBuffer,
    /// Device scratch: DZK output coefficients, (t+1) × 32 B.
    d_dzk_out: DeviceBuffer,
    /// Pinned host buffer for the party-rows D2H (grows lazily).
    host_party_rows: Option<PinnedHostBuffer>,
    host_party_rows_cap: usize,
    /// Dedicated stream for the async H2D of the PRF matrix.
    transfer_stream: Option<CudaStream>,
}

impl AcssGemmCache {
    pub fn new() -> Self {
        Self {
            inv_vand_main: None,
            powers_main: None,
            inv_vand_nonce: None,
            powers_nonce: None,
            key: None,
            d_y_matrix: DeviceBuffer::new(),
            d_nonce_prfs: DeviceBuffer::new(),
            d_party_bytes: DeviceBuffer::new(),
            d_dzk_powers: DeviceBuffer::new(),
            d_dzk_out: DeviceBuffer::new(),
            host_party_rows: None,
            host_party_rows_cap: 0,
            transfer_stream: None,
        }
    }

    /// Build + upload the four cached GEMM contexts for (t, n) if not already
    /// resident. Matrices are exactly the ones the CPU path builds in
    /// `generate_evaluation_points*` (interpolation points {0, 1..t},
    /// evaluation points 1..n, degree t).
    fn ensure_matrices(&mut self, t: usize, n: usize) -> Result<(), String> {
        if self.key == Some((t, n)) && self.inv_vand_main.is_some() {
            return Ok(());
        }

        let mut evaluation_points = Vec::with_capacity(t + 1);
        evaluation_points.push(LargeField::from(0u64));
        for i in 0..t {
            evaluation_points.push(LargeField::from((i + 1) as u64));
        }
        let inv_vand = inverse_vandermonde(vandermonde_matrix(evaluation_points));

        let share_points: Vec<LargeField> = (1..=n).map(|i| LargeField::from(i as u64)).collect();
        let share_powers = powers_matrix(&share_points, t + 1);

        self.inv_vand_main = Some(PreparedFieldGemm::new(&inv_vand)?);
        self.powers_main = Some(PreparedFieldGemm::new(&share_powers)?);
        self.inv_vand_nonce = Some(PreparedFieldGemm::new(&inv_vand)?);
        self.powers_nonce = Some(PreparedFieldGemm::new(&share_powers)?);
        self.key = Some((t, n));
        Ok(())
    }

    fn ensure_host_party_rows(&mut self, needed: usize) -> Result<(), String> {
        if self.host_party_rows.is_none() || self.host_party_rows_cap < needed {
            self.host_party_rows = Some(PinnedHostBuffer::alloc(needed)?);
            self.host_party_rows_cap = needed;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Flattening helpers (host ↔ device layout)
// ---------------------------------------------------------------------------

/// Flatten rows of `LargeField` into the raw 32-byte-per-element CUDA layout
/// (native little-endian limbs — see `protocol::write_elem_bytes`).
fn flatten_rows_le(rows: &[Vec<LargeField>]) -> Vec<u8> {
    if rows.is_empty() {
        return Vec::new();
    }
    let row_elems = rows[0].len();
    let row_bytes = row_elems * FIELD_BYTES;
    let mut flat = vec![0u8; rows.len() * row_bytes];
    flat.par_chunks_mut(row_bytes)
        .zip(rows.par_iter())
        .for_each(|(chunk, row)| {
            for (k, el) in row.iter().enumerate() {
                let src = el as *const LargeField as *const u8;
                // SAFETY: LargeField is a POD 32-byte struct (guarded by the
                // protocol crate's compile-time _LAYOUT_CHECK).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        chunk.as_mut_ptr().add(k * FIELD_BYTES),
                        FIELD_BYTES,
                    );
                }
            }
        });
    flat
}

/// In-place per-u64 byte swap: converts raw native-LE Fp4_61 limbs into the
/// `to_bytes_be` wire layout (and back — the swap is an involution).
fn swap_limbs_le_be(bytes: &mut [u8]) {
    debug_assert_eq!(bytes.len() % LIMB_BYTES, 0);
    for chunk in bytes.chunks_exact_mut(LIMB_BYTES) {
        chunk.reverse();
    }
}

// ---------------------------------------------------------------------------
// Deterministic GPU dealer core
// ---------------------------------------------------------------------------

/// GPU counterpart of `dealer_core_cpu`: same inputs, byte-identical
/// `DealerArtifacts` output (the parity tests assert this). Non-FFT path only.
pub fn dealer_core_gpu(
    secrets: &[LargeField],
    nonce_secret: LargeField,
    blinding_secret: LargeField,
    blinding_nonce_secret: LargeField,
    sec_key_map: HashMap<Replica, Vec<u8>>,
    num_faults: usize,
    num_nodes: usize,
    hash_context: &HashState,
    cache: &mut AcssGemmCache,
) -> Result<DealerArtifacts, String> {
    if secrets.is_empty() {
        return Err("dealer_core_gpu: empty secrets batch".to_string());
    }

    let t = num_faults;
    let n = num_nodes;
    let num_polys = secrets.len();
    let n_coeffs = t + 1;
    let party_stride = num_polys * FIELD_BYTES + FIELD_BYTES;

    let t_start = Instant::now();

    // Phase 1: PRF sampling on CPU — identical inputs/tags to the CPU dealer,
    // so shares are byte-compatible between GPU and CPU dealers.
    let evaluations_prf =
        sample_polynomials_from_prf(secrets.to_vec(), sec_key_map.clone(), t, false, 1u8);
    let t_prf = t_start.elapsed();

    // Phase 2: flatten + async H2D on the transfer stream.
    let y_flat = flatten_rows_le(&evaluations_prf);
    drop(evaluations_prf);
    if cache.transfer_stream.is_none() {
        cache.transfer_stream = Some(CudaStream::create()?);
    }
    {
        let stream = cache.transfer_stream.as_ref().unwrap();
        // SAFETY-free wrapper: DeviceBuffer grows itself; y_flat outlives the
        // synchronize below.
        cache.d_y_matrix.copy_from_host_async(&y_flat, stream)?;
    }

    // Phase 3 (overlapped with the DMA): nonce/blinding/blinding-nonce PRF
    // rows + lazy matrix upload.
    let nonce_prf =
        sample_polynomials_from_prf(vec![nonce_secret], sec_key_map.clone(), t, true, 1u8);
    let blinding_prf =
        sample_polynomials_from_prf(vec![blinding_secret], sec_key_map.clone(), t, true, 2u8);
    let blinding_nonce_prf =
        sample_polynomials_from_prf(vec![blinding_nonce_secret], sec_key_map, t, true, 3u8);
    let nonce_rows: Vec<Vec<LargeField>> = vec![
        nonce_prf.into_iter().next().unwrap(),
        blinding_prf.into_iter().next().unwrap(),
        blinding_nonce_prf.into_iter().next().unwrap(),
    ];
    let nonce_flat = flatten_rows_le(&nonce_rows);
    drop(nonce_rows);

    cache.ensure_matrices(t, n)?;
    cache.transfer_stream.as_ref().unwrap().synchronize()?;
    cache.d_nonce_prfs.copy_from_host(&nonce_flat)?;
    let t_upload = t_start.elapsed();

    // Phase 4: D2D GEMM chains.
    // Main batch: d_coeffs (num_polys × (t+1)), then d_evals (num_polys × n).
    // d_coeffs lives in inv_vand_main's output buffer and must stay untouched
    // until the DZK kernel in phase 8.
    let (d_coeffs, d_evals, d_nonce_coeffs, d_nonce_evals) = {
        let inv_vand_main = cache.inv_vand_main.as_ref().unwrap();
        let powers_main = cache.powers_main.as_ref().unwrap();
        let inv_vand_nonce = cache.inv_vand_nonce.as_ref().unwrap();
        let powers_nonce = cache.powers_nonce.as_ref().unwrap();
        // SAFETY: device pointers sized by the GEMM contracts (batch × K × 32).
        unsafe {
            let d_coeffs = inv_vand_main.multiply_d2d(cache.d_y_matrix.as_ptr(), num_polys)?;
            let d_evals = powers_main.multiply_d2d(d_coeffs, num_polys)?;
            let d_nonce_coeffs = inv_vand_nonce.multiply_d2d(cache.d_nonce_prfs.as_ptr(), 3)?;
            let d_nonce_evals = powers_nonce.multiply_d2d(d_nonce_coeffs, 3)?;
            (d_coeffs, d_evals, d_nonce_coeffs, d_nonce_evals)
        }
    };
    let t_gemm = t_start.elapsed();

    // Phase 5: pack per-party wire rows on device (LE→BE), append the nonce.
    // Velox evaluates all n parties in the GEMM, so t=0 / d_rand_cols=null.
    cache.d_party_bytes.ensure(n * party_stride)?;
    // SAFETY: d_evals is (num_polys × n × 32) B, d_party_bytes n × party_stride,
    // d_nonce_evals row 0 is n × 32 B — all sized above.
    unsafe {
        pack_party_shares(
            std::ptr::null(),
            d_evals,
            cache.d_party_bytes.as_mut_ptr(),
            0,
            n,
            num_polys,
            party_stride,
            FIELD_BYTES,
            LIMB_BYTES,
        )?;
        append_nonce(
            d_nonce_evals,
            cache.d_party_bytes.as_mut_ptr(),
            n,
            party_stride,
            num_polys * FIELD_BYTES,
        )?;
    }

    // Phase 6: D2H the 3 × n nonce evaluations and convert to wire bytes.
    let mut nonce_evals_be = vec![0u8; 3 * n * FIELD_BYTES];
    // SAFETY: d_nonce_evals points to 3 × n × 32 B of GEMM output.
    unsafe { device_to_host(d_nonce_evals, &mut nonce_evals_be, 3 * n * FIELD_BYTES)? };
    swap_limbs_le_be(&mut nonce_evals_be);

    // Phase 7: D2H all packed party rows (pinned) → host commitments.
    // The rows serve both the AES commitments and the outgoing share payloads.
    let total_rows_bytes = n * party_stride;
    cache.ensure_host_party_rows(total_rows_bytes)?;
    let commitments: Vec<Hash> = {
        let host_rows = cache.host_party_rows.as_mut().unwrap();
        cache
            .d_party_bytes
            .copy_to_host(host_rows.as_mut_slice(), total_rows_bytes)?;
        host_rows.as_slice()[..total_rows_bytes]
            .par_chunks(party_stride)
            .map(|row| hash_context.do_hash_aes(row))
            .collect()
    };
    let t_d2h = t_start.elapsed();

    let blinding_commitments: Vec<Hash> = (0..n)
        .map(|i| {
            let blinding: [u8; 32] = nonce_evals_be[(n + i) * FIELD_BYTES..(n + i + 1) * FIELD_BYTES]
                .try_into()
                .unwrap();
            let blinding_nonce: [u8; 32] = nonce_evals_be
                [(2 * n + i) * FIELD_BYTES..(2 * n + i + 1) * FIELD_BYTES]
                .try_into()
                .unwrap();
            hash_context.hash_two(blinding, blinding_nonce)
        })
        .collect();

    // Merkle roots + Fiat-Shamir challenge — identical to the CPU dealer.
    let share_root_comm = MerkleTree::new(commitments.clone(), hash_context).root();
    let blinding_mt_root = MerkleTree::new(blinding_commitments.clone(), hash_context).root();
    let root_comm = hash_context.hash_two(share_root_comm, blinding_mt_root);
    let root_comm_fe = LargeField::from_bytes_be(&root_comm)
        .map_err(|e| format!("root_comm_fe deserialization failed: {:?}", e))?;
    log::info!("Root_comm_fe: {:?}", root_comm_fe);
    let t_hash = t_start.elapsed();

    // Phase 8: DZK coefficient combination on device.
    // powers[j] = r^(j+1), j in 0..num_polys — matches the CPU accumulation.
    let mut powers = Vec::with_capacity(num_polys);
    let mut acc = root_comm_fe.clone();
    for _ in 0..num_polys {
        powers.push(acc.clone());
        acc = acc * root_comm_fe.clone();
    }
    let powers_flat = flatten_rows_le(&[powers]);
    cache.d_dzk_powers.copy_from_host(&powers_flat)?;
    cache.d_dzk_out.ensure(n_coeffs * FIELD_BYTES)?;
    // Blinding coefficients = row 1 of the nonce interpolation output.
    // SAFETY: d_nonce_coeffs holds 3 × (t+1) × 32 B; d_coeffs is still the
    // last output of inv_vand_main (no compute on it since phase 4).
    unsafe {
        let d_blind_coeffs = d_nonce_coeffs.add(n_coeffs * FIELD_BYTES);
        compute_dzk_coeffs(
            d_coeffs,
            d_blind_coeffs,
            cache.d_dzk_powers.as_ptr(),
            cache.d_dzk_out.as_mut_ptr(),
            num_polys,
            n_coeffs,
        )?;
    }
    let mut dzk_be = vec![0u8; n_coeffs * FIELD_BYTES];
    cache.d_dzk_out.copy_to_host(&mut dzk_be, n_coeffs * FIELD_BYTES)?;
    let ser_dzk_coeffs: Vec<LargeFieldSer> =
        dzk_be.chunks(FIELD_BYTES).map(|c| c.to_vec()).collect();
    let t_dzk = t_start.elapsed();

    // Phase 9: split the host rows into per-party artifacts (wire bytes are
    // already big-endian — byte-identical to the CPU dealer's to_bytes_be).
    let rows = &cache.host_party_rows.as_ref().unwrap().as_slice()[..total_rows_bytes];
    let party_wise_shares: Vec<Vec<LargeFieldSer>> = rows
        .par_chunks(party_stride)
        .map(|row| {
            row[..num_polys * FIELD_BYTES]
                .chunks_exact(FIELD_BYTES)
                .map(|c| c.to_vec())
                .collect()
        })
        .collect();
    let nonce_evals_ser: Vec<LargeFieldSer> = (0..n)
        .map(|i| nonce_evals_be[i * FIELD_BYTES..(i + 1) * FIELD_BYTES].to_vec())
        .collect();
    let blinding_nonce_evals_ser: Vec<LargeFieldSer> = (0..n)
        .map(|i| nonce_evals_be[(2 * n + i) * FIELD_BYTES..(2 * n + i + 1) * FIELD_BYTES].to_vec())
        .collect();

    log::info!(
        "[gpu timing] acss dealer: polys={} n={} t={} | prf {:?} | upload {:?} | gemm {:?} | pack+d2h {:?} | hash {:?} | dzk {:?} | total {:?}",
        num_polys,
        n,
        t,
        t_prf,
        t_upload - t_prf,
        t_gemm - t_upload,
        t_d2h - t_gemm,
        t_hash - t_d2h,
        t_dzk - t_hash,
        t_start.elapsed()
    );

    Ok(DealerArtifacts {
        party_wise_shares,
        nonce_evals_ser,
        blinding_nonce_evals_ser,
        commitments,
        blinding_commitments,
        ser_dzk_coeffs,
    })
}

// ---------------------------------------------------------------------------
// Context entry point
// ---------------------------------------------------------------------------

impl Context {
    /// GPU dealer entry (non-FFT only; `init_acss_ab` dispatches here). Falls
    /// back to the CPU dealer on any CUDA error.
    pub async fn init_acss_ab_gpu(&mut self, secrets: Vec<LargeField>, instance_id: usize) {
        let tot_sharings = secrets.len();
        let nonce_secret = rand_field_element();
        let blinding_secret = rand_field_element();
        let blinding_nonce_secret = rand_field_element();

        let gpu_result = dealer_core_gpu(
            &secrets,
            nonce_secret.clone(),
            blinding_secret.clone(),
            blinding_nonce_secret.clone(),
            self.sec_key_map.clone(),
            self.num_faults,
            self.num_nodes,
            &self.hash_context,
            &mut self.gemm_cache,
        );

        let artifacts = match gpu_result {
            Ok(artifacts) => artifacts,
            Err(e) => {
                log::error!(
                    "GPU ACSS dealer failed ({}) — falling back to CPU for instance {}",
                    e,
                    instance_id
                );
                dealer_core_cpu(
                    secrets,
                    nonce_secret,
                    blinding_secret,
                    blinding_nonce_secret,
                    self.sec_key_map.clone(),
                    self.num_faults,
                    self.num_nodes,
                    &self.hash_context,
                )
                .await
            }
        };

        self.broadcast_dealer_artifacts(instance_id, tot_sharings, artifacts)
            .await;
    }
}
