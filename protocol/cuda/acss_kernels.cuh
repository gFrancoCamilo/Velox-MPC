// acss_kernels.cuh — ACSS-specific GPU kernels for the fused pipeline.
//
// These kernels bridge the gaps between GEMM output, AES/Merkle input, and
// DZK input, keeping all intermediate data resident on the device.
//
// Vendored from async_mpc/secret_sharing/acss_ab/cuda/acss_kernels.cuh with
// one protocol adaptation: Velox's DZK proof is a broadcast *coefficient*
// vector (a random linear combination of the sharing polynomials'
// coefficients), not per-party evaluations, so acss_compute_dzk_evals is
// replaced by acss_compute_dzk_coeffs.
//
// Include this header from the Rust build only via the corresponding .cu file.

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Kernel A: Pack per-party share bytes
//
// Combines PRF-masked shares (rand_cols, on device) and polynomial evaluation
// results (evals_gemm, on device) into per-party flat byte arrays ready to be
// hashed into commitments.
//
// d_rand_cols  : (t × num_polys) field elements, small_field_bytes each.
//                Layout: row-major [party_idx < t][poly_idx]. May be NULL when
//                t == 0 (Velox evaluates all n parties in the GEMM, so the
//                PRF split is unused and every party takes the evals branch).
// d_evals_gemm : (num_polys × n_eval_parties) field elements, small_field_bytes each.
//                Layout: row-major [poly_idx][eval_party_idx].
//                Output from the powers GEMM — poly is the outer dimension.
// d_out        : (n_parties × party_stride) bytes.
//                Row party_idx contains: [share0 || share1 || ... || share_{num_polys-1}]
//                The nonce is appended separately by acss_append_nonce.
// t            : number of PRF-masked parties (rows in d_rand_cols).
// n_eval_parties: n_parties - t.
// num_polys    : number of polynomials (= number of secrets being shared).
// party_stride : bytes per party row in d_out (must be >= num_polys * small_field_bytes + 32).
// small_field_bytes : width of one share element (32 for Velox's Fp4_61 LargeField).
// limb_bytes   : base-prime width for the per-limb LE→BE swap (8 for the M61 family).
// ---------------------------------------------------------------------------
int acss_pack_party_shares(
    const uint8_t* d_rand_cols,
    const uint8_t* d_evals_gemm,
    uint8_t* d_out,
    int t, int n_eval_parties, int num_polys, int party_stride,
    int small_field_bytes, int limb_bytes
);

// ---------------------------------------------------------------------------
// Kernel B: Append nonce evaluations to party byte arrays
//
// Appends the 32-byte nonce polynomial evaluation for each party to the end of
// their row in d_out (at offset num_polys * small_field_bytes).
//
// d_nonce_evals: Flat byte array, party_idx in 0..n_parties, each 32 bytes of
//               raw native Fp4_61 (GEMM output); converted per-u64 LE→BE on
//               store so the row matches the verifier's to_bytes_be layout.
// d_out        : (n_parties × party_stride) bytes (same as in acss_pack_party_shares).
// nonce_offset : byte offset within each party's row where the nonce goes
//               (= num_polys * small_field_bytes).
// ---------------------------------------------------------------------------
int acss_append_nonce(
    const uint8_t* d_nonce_evals,
    uint8_t* d_out,
    int n_parties, int party_stride, int nonce_offset
);

// ---------------------------------------------------------------------------
// Kernel C: Build blinding commitment hash inputs on device
//
// d_nonce_evals: (3 × n_parties × 32) bytes.
// d_one/d_two  : (n_parties × 32) bytes each, interpreted as 8 u32 words
//                per party by aes_hash_batch_d2d.
//
// Dormant: kept for the future GPU-Merkle path (see acss_merkle_batched_d2d).
// ---------------------------------------------------------------------------
int acss_build_blinding_hash_inputs(
    const uint8_t* d_nonce_evals,
    uint8_t* d_one,
    uint8_t* d_two,
    int n_parties
);

// ---------------------------------------------------------------------------
// Kernel D: Compute DZK coefficient vector on device
//
// dzk[c] = blind_coeffs[c] + Σ_j coeffs[j][c] · powers[j]
//
// This is the coefficient-space random linear combination Velox's dealer
// broadcasts (init_acss_ab's `dzk_coeffs`), with powers[j] = r^(j+1) for the
// Fiat-Shamir challenge r derived from the commitment Merkle roots.
//
// d_coeffs       : (num_polys × n_coeffs) raw native Fp4_61 — the interpolation
//                  GEMM output, element (j, c) at d_coeffs[j * n_coeffs + c].
// d_blind_coeffs : n_coeffs raw native Fp4_61 — the blinding polynomial's
//                  coefficients (row 1 of the nonce-batch interpolation GEMM).
// d_powers       : num_polys raw native Fp4_61 challenge powers.
// d_out          : (n_coeffs × 32) big-endian Fp4_61 wire bytes.
// ---------------------------------------------------------------------------
int acss_compute_dzk_coeffs(
    const uint8_t* d_coeffs,
    const uint8_t* d_blind_coeffs,
    const uint8_t* d_powers,
    uint8_t* d_out,
    int num_polys,
    int n_coeffs
);

// ---------------------------------------------------------------------------
// Driver: Batched Merkle tree reduction, fully on device
//
// Computes one 32-byte Merkle root per party from their packed byte data.
// Uses aes_hash_batch_d2d (from gpu_aes_wrapper.cu) for each tree level, so
// the intermediate hashes never leave the GPU.
//
// Dormant: the live dealer hashes commitments on the host with do_hash_aes
// (matching async_mpc's correctness fix); this driver is retained for a
// future GPU-Merkle revival.
//
// aes_ctx      : Opaque AESContext* (from aes_init/aes_set_key).
// d_party_data : (n_parties × party_stride) bytes, each party's leaf data.
// d_roots_out  : (n_parties × 32) bytes output — one 32-byte root per party.
//
// Returns 0 on success, -1 on CUDA error.
// ---------------------------------------------------------------------------
int acss_merkle_batched_d2d(
    void* aes_ctx,
    const uint8_t* d_party_data,
    int n_parties,
    int party_stride,
    uint8_t* d_roots_out
);

// Returns a pointer to a static C string describing the last ACSS CUDA error.
const char* acss_get_last_error(void);

#ifdef __cplusplus
}
#endif
