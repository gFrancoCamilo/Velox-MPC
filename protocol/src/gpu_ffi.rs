//! CUDA FFI bridge for the Mersenne-61 family GEMM kernel.
//!
//! Compiled only with `--features gpu`. On Mac / non-CUDA hosts this module is
//! gated out and the protocol crate builds pure-Rust.
//!
//! Mirrors async_mpc's `gpu_ffi.rs` lifecycle:
//!   `CudaFieldGEMM::new()` → `set_matrix(V, K, N, field_id)`
//!   → `compute(S, batch)` → drop (calls `field_gemm_free`).
//!
//! Velox's protocol field is currently `Mersenne61Degree4ExtensionField`
//! (32-byte Fp4 elements). The C wrapper also handles the smaller M61_FP /
//! M61_FP2 cases for completeness — `resolve_field` dispatches on `TypeId`.

use std::any::TypeId;
use std::os::raw::{c_int, c_uchar, c_uint};

use crate::mersenne_61::{
    Mersenne61Degree2ExtensionField, Mersenne61Degree4ExtensionField, Mersenne61Field,
};
use crate::LargeField;

/// Opaque CUDA-side context.
#[repr(C)]
pub struct FieldGEMMContext {
    _private: [u8; 0],
}

/// Field tag, kept binary-compatible with async_mpc's `GpuFieldId`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuFieldId {
    M61Fp = 4,
    M61Fp2 = 5,
    M61Fp4 = 6,
}

extern "C" {
    fn field_gemm_init() -> *mut FieldGEMMContext;
    fn field_gemm_set_matrix(
        ctx: *mut FieldGEMMContext,
        v_bytes: *const c_uchar,
        k: c_uint,
        n: c_uint,
        field_id: c_int,
    ) -> c_int;
    fn field_gemm_compute(
        ctx: *mut FieldGEMMContext,
        s_bytes: *const c_uchar,
        batch: c_uint,
        o_bytes: *mut c_uchar,
    ) -> c_int;
    fn field_gemm_compute_d2d(
        ctx: *mut FieldGEMMContext,
        d_s_bytes: *const c_uchar,
        batch: c_uint,
    ) -> c_int;
    fn field_gemm_get_output_device(ctx: *mut FieldGEMMContext) -> *const c_uchar;
    fn field_gemm_free(ctx: *mut FieldGEMMContext);
}

/// Resolve the GPU field tag + per-element byte size for a Rust field type.
/// Panics on unsupported fields — matches async_mpc's behaviour.
fn resolve_field<F: 'static>() -> (GpuFieldId, usize) {
    let id = TypeId::of::<F>();
    if id == TypeId::of::<Mersenne61Field>() {
        (GpuFieldId::M61Fp, 8)
    } else if id == TypeId::of::<Mersenne61Degree2ExtensionField>() {
        (GpuFieldId::M61Fp2, 16)
    } else if id == TypeId::of::<Mersenne61Degree4ExtensionField>() {
        (GpuFieldId::M61Fp4, 32)
    } else {
        panic!("gpu_ffi: unsupported field type for GPU GEMM");
    }
}

/// Safe wrapper. `Drop` releases device buffers.
pub struct CudaFieldGEMM {
    ctx: *mut FieldGEMMContext,
    k: u32,
    n: u32,
    field_id: GpuFieldId,
    elem_size: usize,
}

impl CudaFieldGEMM {
    pub fn new(field_id: GpuFieldId, elem_size: usize) -> Result<Self, String> {
        // SAFETY: field_gemm_init returns NULL on failure.
        let ctx = unsafe { field_gemm_init() };
        if ctx.is_null() {
            return Err("field_gemm_init returned NULL (CUDA device unavailable?)".to_string());
        }
        Ok(Self { ctx, k: 0, n: 0, field_id, elem_size })
    }

    /// Upload a K×N V matrix.
    pub fn set_matrix(&mut self, v_bytes: &[u8], k: usize, n: usize) -> Result<(), String> {
        let expected = k
            .checked_mul(n)
            .and_then(|x| x.checked_mul(self.elem_size))
            .ok_or_else(|| "K*N overflow".to_string())?;
        if v_bytes.len() != expected {
            return Err(format!(
                "set_matrix: expected {} bytes for K={}, N={}, elem_size={}; got {}",
                expected, k, n, self.elem_size, v_bytes.len()
            ));
        }
        // SAFETY: pointer valid for v_bytes's lifetime; kernel copies before returning.
        let rc = unsafe {
            field_gemm_set_matrix(
                self.ctx,
                v_bytes.as_ptr(),
                k as c_uint,
                n as c_uint,
                self.field_id as c_int,
            )
        };
        if rc != 0 {
            return Err(format!("field_gemm_set_matrix failed (rc={})", rc));
        }
        self.k = k as u32;
        self.n = n as u32;
        Ok(())
    }

    /// Run the GEMM. Output buffer is `batch * N * elem_size` bytes.
    pub fn compute(&self, s_bytes: &[u8], batch: usize) -> Result<Vec<u8>, String> {
        let k = self.k as usize;
        let n = self.n as usize;
        let expected_in = batch
            .checked_mul(k)
            .and_then(|x| x.checked_mul(self.elem_size))
            .ok_or_else(|| "batch*K overflow".to_string())?;
        if s_bytes.len() != expected_in {
            return Err(format!(
                "compute: expected {} bytes for batch={}, K={}, elem_size={}; got {}",
                expected_in, batch, k, self.elem_size, s_bytes.len()
            ));
        }
        let out_len = batch
            .checked_mul(n)
            .and_then(|x| x.checked_mul(self.elem_size))
            .ok_or_else(|| "batch*N overflow".to_string())?;
        let mut out = vec![0u8; out_len];
        // SAFETY: both buffers are valid for their advertised lengths.
        let rc = unsafe {
            field_gemm_compute(self.ctx, s_bytes.as_ptr(), batch as c_uint, out.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!("field_gemm_compute failed (rc={})", rc));
        }
        Ok(out)
    }

    /// Run the GEMM with `d_s` already resident on the device; the result stays
    /// in this context's device output buffer and its pointer is returned.
    ///
    /// The returned pointer is valid until the next `compute*`/`set_matrix`
    /// call or drop **on this context** — chain GEMMs across *different*
    /// contexts, never re-use one before its output is consumed.
    ///
    /// # Safety
    /// `d_s` must be a valid CUDA device pointer to `batch * K * elem_size` bytes.
    pub unsafe fn compute_d2d(&self, d_s: *const u8, batch: usize) -> Result<*const u8, String> {
        let rc = field_gemm_compute_d2d(self.ctx, d_s, batch as c_uint);
        if rc != 0 {
            return Err(format!("field_gemm_compute_d2d failed (rc={})", rc));
        }
        let out = field_gemm_get_output_device(self.ctx);
        if out.is_null() {
            return Err("field_gemm_get_output_device returned NULL".to_string());
        }
        Ok(out)
    }

    /// Raw device pointer to the last GEMM output (see `compute_d2d` lifetime).
    pub fn device_output_ptr(&self) -> *const u8 {
        unsafe { field_gemm_get_output_device(self.ctx) }
    }
}

impl Drop for CudaFieldGEMM {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { field_gemm_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// A GEMM context with its V matrix uploaded once and kept resident on device,
/// for repeated `multiply_d2d` calls against the same matrix. This is the
/// cache unit for the GPU ACSS dealer (mirrors async_mpc's `PreparedFieldGemm`).
///
/// Semantics match `gpu_matrix_matrix_multiply`: for a `matrix` of R rows × C
/// cols, `multiply_d2d(d_s, batch)` computes `output[i][r] = Σ_l matrix[r][l]
/// * S[i][l]` where `d_s` holds `batch` row-major vectors of length C, i.e.
/// batch-stacked (matrix · vector_i). Output is (batch × R) row-major Fp4_61
/// on device.
pub struct PreparedFieldGemm {
    gemm: CudaFieldGEMM,
    /// Inner (contraction) dimension = the matrix's column count C.
    pub k: usize,
    /// Output dimension = the matrix's row count R.
    pub n: usize,
}

impl PreparedFieldGemm {
    /// Upload `matrix` (R × C, row-major) transposed into device memory.
    pub fn new(matrix: &[Vec<LargeField>]) -> Result<Self, String> {
        let r = matrix.len();
        if r == 0 {
            return Err("PreparedFieldGemm: empty matrix".to_string());
        }
        let c = matrix[0].len();
        if matrix.iter().any(|row| row.len() != c) {
            return Err("PreparedFieldGemm: ragged matrix".to_string());
        }

        let (field_id, elem_size) = resolve_field::<Mersenne61Degree4ExtensionField>();
        // V[l, r] = matrix[r][l] so the kernel's O = S × V contracts over C.
        let mut v_bytes = Vec::with_capacity(c * r * elem_size);
        for l in 0..c {
            for r_idx in 0..r {
                write_elem_bytes(&matrix[r_idx][l], &mut v_bytes);
            }
        }

        let mut gemm = CudaFieldGEMM::new(field_id, elem_size)?;
        gemm.set_matrix(&v_bytes, c, r)?;
        Ok(Self { gemm, k: c, n: r })
    }

    /// Device-to-device multiply; see `CudaFieldGEMM::compute_d2d` for the
    /// output-pointer lifetime contract.
    ///
    /// # Safety
    /// `d_s` must be a valid CUDA device pointer to `batch * k * 32` bytes.
    pub unsafe fn multiply_d2d(&self, d_s: *const u8, batch: usize) -> Result<*const u8, String> {
        self.gemm.compute_d2d(d_s, batch)
    }

    /// Host-to-host multiply against the resident matrix (upload S, download O).
    pub fn multiply(&self, s_bytes: &[u8], batch: usize) -> Result<Vec<u8>, String> {
        self.gemm.compute(s_bytes, batch)
    }

    pub fn device_output_ptr(&self) -> *const u8 {
        self.gemm.device_output_ptr()
    }
}

// SAFETY: the wrapped context pointer is only ever passed to CUDA runtime
// calls; nothing is dereferenced on the host.
unsafe impl Send for PreparedFieldGemm {}
unsafe impl Sync for PreparedFieldGemm {}

/// Serialise a `LargeField` (Fp4_61) into the 32-byte CUDA layout. Lambdaworks
/// stores `FieldElement<Fp4_61>` as `[Fp2E; 2]` = `[[FpE; 2]; 2]` of plain u64s
/// in their canonical representative — byte-identical to the CUDA `Fp4_61`
/// struct, so a raw memcpy is correct.
pub fn write_elem_bytes(elem: &LargeField, buf: &mut Vec<u8>) {
    let ptr = elem as *const LargeField as *const u8;
    // SAFETY: LargeField is POD; size guarded by the LAYOUT_CHECK below.
    buf.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, 32) });
}

/// Inverse of `write_elem_bytes`: reinterpret 32 CUDA-layout bytes as a
/// `LargeField`.
pub fn read_elem_bytes(bytes: &[u8]) -> LargeField {
    debug_assert_eq!(bytes.len(), 32);
    unsafe {
        let mut elem = std::mem::MaybeUninit::<LargeField>::uninit();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), elem.as_mut_ptr() as *mut u8, 32);
        elem.assume_init()
    }
}

/// Compile-time guard: layout assumption. If lambdaworks ever changes the
/// Mersenne-61 Fp4 representation this fails the build instead of silently
/// corrupting GPU output.
const _LAYOUT_CHECK: () = {
    assert!(std::mem::size_of::<LargeField>() == 32);
};

/// GPU implementation of `matrix_matrix_multiply(matrix, vectors, row_major)`.
///
/// Layout contract is identical to the CPU path. The C wrapper's GEMM computes
/// `O = S × V` of shape (batch × N), where:
///   - V = velox's `matrix` argument, treated as (K × N) = (cols × rows-of-V).
///   - S = velox's `vectors` argument, treated as (batch × K).
/// In velox's terms: `matrix` is R rows × C cols; each of the K `vectors` is
/// length C. We compute `output[i][r] = Σ_l matrix[r][l] * vectors[i][l]`,
/// i.e. K-stacked (matrix · vector_i) — matching the CPU semantics.
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

    let (field_id, elem_size) = resolve_field::<Mersenne61Degree4ExtensionField>();
    debug_assert_eq!(elem_size, 32);

    // Pack V as (K_inner × N) row-major where K_inner = C and N = R.
    // The kernel computes O = S × V, so we want V[l, r] = matrix[r][l] (transposed),
    // then output[i, r] = Σ_l S[i, l] * V[l, r] = Σ_l vectors[i][l] * matrix[r][l].
    let mut v_bytes = Vec::with_capacity(c * r * elem_size);
    for l in 0..c {
        for r_idx in 0..r {
            write_elem_bytes(&matrix[r_idx][l], &mut v_bytes);
        }
    }
    // S is (batch × K_inner) = (K × C) row-major.
    let mut s_bytes = Vec::with_capacity(k * c * elem_size);
    for vec in vectors {
        for elem in vec {
            write_elem_bytes(elem, &mut s_bytes);
        }
    }

    let mut gemm = match CudaFieldGEMM::new(field_id, elem_size) {
        Ok(g) => g,
        Err(e) => {
            log::error!("CUDA init failed: {} — falling back to CPU GEMM", e);
            return crate::poly::matrix_matrix_multiply_cpu(matrix, vectors, row_major);
        }
    };
    if let Err(e) = gemm.set_matrix(&v_bytes, c, r) {
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

    // Output bytes are (batch × N) = (K × R) row-major: chunks of R elements per vector.
    // That's `output[i][r_idx]` natively → matches `row_major == false` in our convention.
    let mut out_kr: Vec<Vec<LargeField>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut row = Vec::with_capacity(r);
        for r_idx in 0..r {
            let off = (i * r + r_idx) * elem_size;
            row.push(read_elem_bytes(&o_bytes[off..off + elem_size]));
        }
        out_kr.push(row);
    }

    if row_major {
        crate::poly::transpose(out_kr)
    } else {
        out_kr
    }
}
