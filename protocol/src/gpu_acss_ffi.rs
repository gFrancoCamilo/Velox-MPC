//! Rust FFI bindings and safe wrappers for the ACSS-specific CUDA kernels
//! compiled from `cuda/acss_kernels.cu`.
//!
//! Compiled only with `--features gpu`. Adapted from
//! async_mpc/secret_sharing/acss_ab/src/protocol/acss_kernels_ffi.rs; the
//! per-party DZK evaluation wrapper is replaced by `compute_dzk_coeffs`
//! (Velox broadcasts a DZK coefficient vector instead).

extern "C" {
    /// Returns a pointer to a static C string describing the last ACSS CUDA error.
    fn acss_get_last_error() -> *const std::ffi::c_char;

    /// See cuda/acss_kernels.cuh for full parameter documentation.
    fn acss_pack_party_shares(
        d_rand_cols: *const u8,
        d_evals_gemm: *const u8,
        d_out: *mut u8,
        t: i32,
        n_eval_parties: i32,
        num_polys: i32,
        party_stride: i32,
        small_field_bytes: i32,
        limb_bytes: i32,
    ) -> i32;

    fn acss_append_nonce(
        d_nonce_evals: *const u8,
        d_out: *mut u8,
        n_parties: i32,
        party_stride: i32,
        nonce_offset: i32,
    ) -> i32;

    fn acss_build_blinding_hash_inputs(
        d_nonce_evals: *const u8,
        d_one: *mut u8,
        d_two: *mut u8,
        n_parties: i32,
    ) -> i32;

    fn acss_compute_dzk_coeffs(
        d_coeffs: *const u8,
        d_blind_coeffs: *const u8,
        d_powers: *const u8,
        d_out: *mut u8,
        num_polys: i32,
        n_coeffs: i32,
    ) -> i32;

    fn acss_merkle_batched_d2d(
        aes_ctx: *mut u8,
        d_party_data: *const u8,
        n_parties: i32,
        party_stride: i32,
        d_roots_out: *mut u8,
    ) -> i32;
}

/// Retrieve the last error message set by an ACSS CUDA kernel.
pub fn get_acss_last_error() -> String {
    unsafe {
        let ptr = acss_get_last_error();
        if ptr.is_null() {
            return "unknown ACSS error".to_string();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Pack per-party share bytes from GEMM evals (and optional PRF rand_cols)
/// into per-party rows in `d_out`, converting each base-field limb to the
/// big-endian wire layout.
///
/// Velox calls this with `t = 0`, `n_eval_parties = n`, and a null
/// `d_rand_cols` — every party's shares come from the evaluation GEMM.
///
/// # Safety
/// All device pointers must be valid for the sizes documented in
/// cuda/acss_kernels.cuh.
pub unsafe fn pack_party_shares(
    d_rand_cols: *const u8,
    d_evals_gemm: *const u8,
    d_out: *mut u8,
    t: usize,
    n_eval_parties: usize,
    num_polys: usize,
    party_stride: usize,
    small_field_bytes: usize,
    limb_bytes: usize,
) -> Result<(), String> {
    let rc = acss_pack_party_shares(
        d_rand_cols,
        d_evals_gemm,
        d_out,
        t as i32,
        n_eval_parties as i32,
        num_polys as i32,
        party_stride as i32,
        small_field_bytes as i32,
        limb_bytes as i32,
    );
    if rc != 0 {
        Err(format!(
            "acss_pack_party_shares failed: {}",
            get_acss_last_error()
        ))
    } else {
        Ok(())
    }
}

/// Append nonce evaluations (32 raw native Fp4_61 bytes each, converted to
/// big-endian on store) to each party's row in `d_out`.
///
/// # Safety
/// All device pointers must be valid for the documented sizes.
pub unsafe fn append_nonce(
    d_nonce_evals: *const u8,
    d_out: *mut u8,
    n_parties: usize,
    party_stride: usize,
    nonce_offset: usize,
) -> Result<(), String> {
    let rc = acss_append_nonce(
        d_nonce_evals,
        d_out,
        n_parties as i32,
        party_stride as i32,
        nonce_offset as i32,
    );
    if rc != 0 {
        Err(format!(
            "acss_append_nonce failed: {}",
            get_acss_last_error()
        ))
    } else {
        Ok(())
    }
}

/// Build D2D AES inputs for blinding commitments (dormant — GPU-Merkle path).
///
/// # Safety
/// All device pointers must be valid and large enough.
#[allow(dead_code)]
pub unsafe fn build_blinding_hash_inputs(
    d_nonce_evals: *const u8,
    d_one: *mut u8,
    d_two: *mut u8,
    n_parties: usize,
) -> Result<(), String> {
    let rc = acss_build_blinding_hash_inputs(d_nonce_evals, d_one, d_two, n_parties as i32);
    if rc != 0 {
        Err(format!(
            "acss_build_blinding_hash_inputs failed: {}",
            get_acss_last_error()
        ))
    } else {
        Ok(())
    }
}

/// Compute the DZK coefficient vector on device:
/// `dzk[c] = blind_coeffs[c] + Σ_j coeffs[j][c] · powers[j]`.
/// Inputs are raw native Fp4_61 GEMM output; `d_out` receives `n_coeffs × 32`
/// big-endian wire bytes.
///
/// # Safety
/// All device pointers must be valid for the documented sizes; `d_coeffs`
/// must still hold the interpolation GEMM output (do not reuse its GEMM
/// context before this call).
pub unsafe fn compute_dzk_coeffs(
    d_coeffs: *const u8,
    d_blind_coeffs: *const u8,
    d_powers: *const u8,
    d_out: *mut u8,
    num_polys: usize,
    n_coeffs: usize,
) -> Result<(), String> {
    let rc = acss_compute_dzk_coeffs(
        d_coeffs,
        d_blind_coeffs,
        d_powers,
        d_out,
        num_polys as i32,
        n_coeffs as i32,
    );
    if rc != 0 {
        Err(format!(
            "acss_compute_dzk_coeffs failed: {}",
            get_acss_last_error()
        ))
    } else {
        Ok(())
    }
}

/// Run a batched Merkle tree reduction on device (dormant — GPU-Merkle path).
///
/// `aes_ctx_ptr` must be the raw pointer from `CudaAES::ctx_mut_ptr`.
///
/// # Safety
/// All device pointers must be valid and large enough.
#[allow(dead_code)]
pub unsafe fn merkle_batched_d2d(
    aes_ctx_ptr: *mut u8,
    d_party_data: *const u8,
    n_parties: usize,
    party_stride: usize,
    d_roots_out: *mut u8,
) -> Result<(), String> {
    let rc = acss_merkle_batched_d2d(
        aes_ctx_ptr,
        d_party_data,
        n_parties as i32,
        party_stride as i32,
        d_roots_out,
    );
    if rc != 0 {
        Err(format!(
            "acss_merkle_batched_d2d failed: {}",
            get_acss_last_error()
        ))
    } else {
        Ok(())
    }
}
