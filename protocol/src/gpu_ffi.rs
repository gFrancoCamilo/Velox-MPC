//! CUDA FFI bridge for BN254 matrix-matrix multiply.
//!
//! Compiled only with `--features gpu`. On Mac / non-CUDA hosts this module is
//! gated out and the protocol crate builds pure-Rust.
//!
//! Mirrors the lifecycle pattern of `async_mpc/fields/src/gpu_ffi.rs`:
//!   `CudaBn254Gemm::new()`  →  `set_matrix(V, K, N)`  →  `compute(S, batch)`
//!   →  drop (calls `bn254_gemm_free`).
//!
//! Phase A: the C ABI links and `init`/`free` work; `compute` returns a
//! sentinel error until Phases B–C land the actual kernel.

use std::os::raw::{c_int, c_uchar, c_uint};

use crate::LargeField;

/// Opaque CUDA-side context. Never inspected from Rust.
#[repr(C)]
pub struct Bn254GemmContext {
    _private: [u8; 0],
}

extern "C" {
    fn bn254_gemm_init() -> *mut Bn254GemmContext;
    fn bn254_gemm_set_matrix(
        ctx: *mut Bn254GemmContext,
        v_bytes: *const c_uchar,
        k: c_uint,
        n: c_uint,
    ) -> c_int;
    fn bn254_gemm_compute(
        ctx: *mut Bn254GemmContext,
        s_bytes: *const c_uchar,
        batch: c_uint,
        o_bytes: *mut c_uchar,
    ) -> c_int;
    fn bn254_gemm_free(ctx: *mut Bn254GemmContext);
}

/// Bytes per BN254 field element (4 × u64, Montgomery form).
pub const BN254_ELEM_BYTES: usize = 32;

/// Compile-time guard: `LargeField` MUST have the same byte layout the CUDA
/// kernel expects. If lambdaworks ever changes the BN254 representation this
/// fails to compile loud-and-early instead of silently corrupting GPU output.
const _LAYOUT_CHECK: () = {
    assert!(std::mem::size_of::<LargeField>() == BN254_ELEM_BYTES);
};

/// Safe wrapper around the CUDA context. `Drop` releases device buffers.
pub struct CudaBn254Gemm {
    ctx: *mut Bn254GemmContext,
    k: u32,
    n: u32,
}

// Raw pointer is not auto-Send/Sync; the CUDA context is owned by this struct
// and freed on Drop, but we don't expose it across threads in the wrapper.
// (If/when we ever want cross-thread use we'd guard with a Mutex.)

impl CudaBn254Gemm {
    pub fn new() -> Result<Self, String> {
        // SAFETY: bn254_gemm_init returns NULL on failure.
        let ctx = unsafe { bn254_gemm_init() };
        if ctx.is_null() {
            return Err("bn254_gemm_init returned NULL (CUDA device unavailable?)".to_string());
        }
        Ok(Self { ctx, k: 0, n: 0 })
    }

    /// Upload the K×N V matrix. `v_bytes` must be `K*N*BN254_ELEM_BYTES` long.
    pub fn set_matrix(&mut self, v_bytes: &[u8], k: usize, n: usize) -> Result<(), String> {
        let expected = k.checked_mul(n)
            .and_then(|x| x.checked_mul(BN254_ELEM_BYTES))
            .ok_or_else(|| "K*N overflow".to_string())?;
        if v_bytes.len() != expected {
            return Err(format!(
                "set_matrix: expected {} bytes for K={}, N={}; got {}",
                expected, k, n, v_bytes.len()
            ));
        }
        // SAFETY: pointer is valid for the lifetime of v_bytes; the kernel copies it.
        let rc = unsafe {
            bn254_gemm_set_matrix(self.ctx, v_bytes.as_ptr(), k as c_uint, n as c_uint)
        };
        if rc != 0 {
            return Err(format!("bn254_gemm_set_matrix failed (rc={})", rc));
        }
        self.k = k as u32;
        self.n = n as u32;
        Ok(())
    }

    /// Run the GEMM. `s_bytes` is `batch*N*BN254_ELEM_BYTES`; output is `batch*K*BN254_ELEM_BYTES`.
    pub fn compute(&self, s_bytes: &[u8], batch: usize) -> Result<Vec<u8>, String> {
        let n = self.n as usize;
        let k = self.k as usize;
        let expected_in = batch.checked_mul(n)
            .and_then(|x| x.checked_mul(BN254_ELEM_BYTES))
            .ok_or_else(|| "batch*N overflow".to_string())?;
        if s_bytes.len() != expected_in {
            return Err(format!(
                "compute: expected {} bytes for batch={}, N={}; got {}",
                expected_in, batch, n, s_bytes.len()
            ));
        }
        let out_len = batch
            .checked_mul(k)
            .and_then(|x| x.checked_mul(BN254_ELEM_BYTES))
            .ok_or_else(|| "batch*K overflow".to_string())?;
        let mut out = vec![0u8; out_len];
        // SAFETY: both buffers are valid for their advertised lengths.
        let rc = unsafe {
            bn254_gemm_compute(self.ctx, s_bytes.as_ptr(), batch as c_uint, out.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!(
                "bn254_gemm_compute failed (rc={}); kernel may not be implemented yet",
                rc
            ));
        }
        Ok(out)
    }
}

impl Drop for CudaBn254Gemm {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx came from bn254_gemm_init and is freed at most once.
            unsafe { bn254_gemm_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// Serialise a `LargeField` to its 32-byte CUDA representation. Identical to
/// the in-memory layout of `FieldElement<BN254PrimeField>` (Montgomery, 4×u64
/// little-endian limbs). Mirrors `async_mpc/fields/src/poly.rs::write_elem_bytes`.
fn write_elem_bytes(elem: &LargeField, buf: &mut Vec<u8>) {
    let ptr = elem as *const LargeField as *const u8;
    // SAFETY: LargeField is POD; size guarded by _LAYOUT_CHECK above.
    buf.extend_from_slice(unsafe {
        std::slice::from_raw_parts(ptr, BN254_ELEM_BYTES)
    });
}

/// Inverse of `write_elem_bytes`. Bytes must have come from the CUDA kernel or
/// from `write_elem_bytes` itself.
fn read_elem_bytes(bytes: &[u8]) -> LargeField {
    debug_assert_eq!(bytes.len(), BN254_ELEM_BYTES);
    // SAFETY: bytes is 32 B of a Montgomery-form BN254 element; we copy into
    // an uninitialised LargeField slot and bless it.
    unsafe {
        let mut elem = std::mem::MaybeUninit::<LargeField>::uninit();
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            elem.as_mut_ptr() as *mut u8,
            BN254_ELEM_BYTES,
        );
        elem.assume_init()
    }
}

/// GPU implementation of `matrix_matrix_multiply(matrix, vectors, row_major)`.
///
/// Layout contract identical to the CPU path (see `poly.rs::matrix_matrix_multiply_cpu`):
///   matrix: R×C (R rows of C elements each)
///   vectors: K vectors each of length C
///   row_major=true  → output[r][i] = Σ_l matrix[r][l] * vectors[i][l] (shape R×K)
///   row_major=false → output transposed to shape K×R
pub fn gpu_matrix_matrix_multiply(
    matrix: &[Vec<LargeField>],
    vectors: &[Vec<LargeField>],
    row_major: bool,
) -> Vec<Vec<LargeField>> {
    let r = matrix.len();
    let k = vectors.len();
    if r == 0 || k == 0 {
        return Vec::new();
    }
    let c = matrix[0].len();
    if matrix.iter().any(|row| row.len() != c) || vectors.iter().any(|v| v.len() != c) {
        log::error!("gpu_matrix_matrix_multiply: ragged input");
        return Vec::new();
    }

    let mut v_bytes = Vec::with_capacity(r * c * BN254_ELEM_BYTES);
    for row in matrix {
        for elem in row {
            write_elem_bytes(elem, &mut v_bytes);
        }
    }
    let mut s_bytes = Vec::with_capacity(k * c * BN254_ELEM_BYTES);
    for vec in vectors {
        for elem in vec {
            write_elem_bytes(elem, &mut s_bytes);
        }
    }

    let mut gemm = match CudaBn254Gemm::new() {
        Ok(g) => g,
        Err(e) => {
            log::error!("CUDA init failed: {} — falling back to CPU GEMM", e);
            return crate::poly::matrix_matrix_multiply_cpu(matrix, vectors, row_major);
        }
    };
    if let Err(e) = gemm.set_matrix(&v_bytes, r, c) {
        log::error!("set_matrix failed: {} — falling back to CPU GEMM", e);
        return crate::poly::matrix_matrix_multiply_cpu(matrix, vectors, row_major);
    }
    let o_bytes = match gemm.compute(&s_bytes, k) {
        Ok(b) => b,
        Err(e) => {
            log::error!("compute failed: {} — falling back to CPU GEMM", e);
            return crate::poly::matrix_matrix_multiply_cpu(matrix, vectors, row_major);
        }
    };
    drop(gemm);

    // Output bytes are batch×K = k×r row-major: chunks of K elements per vector.
    // That natively yields output[i][r], i.e. row_major=false in our convention.
    let mut out_kr: Vec<Vec<LargeField>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut row = Vec::with_capacity(r);
        for r_idx in 0..r {
            let off = (i * r + r_idx) * BN254_ELEM_BYTES;
            row.push(read_elem_bytes(&o_bytes[off..off + BN254_ELEM_BYTES]));
        }
        out_kr.push(row);
    }

    if row_major {
        // Want output[r][i]; transpose the K×R buffer above.
        crate::poly::transpose(out_kr)
    } else {
        out_kr
    }
}
