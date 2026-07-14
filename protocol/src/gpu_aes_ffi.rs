//! Rust wrapper for the CUDA AES hash context (`gpu_aes_wrapper.cu`).
//!
//! Compiled only with `--features gpu`. Trimmed from async_mpc's
//! crypto/src/lib.rs (`CudaAES`).
//!
//! Status: experimental / dormant. The live GPU ACSS dealer hashes
//! commitments on the host with `HashState::do_hash_aes` (mirroring
//! async_mpc's correctness fix — the fully-GPU Merkle path produced
//! mismatching roots there). This context exists to drive
//! `acss_merkle_batched_d2d` if that path is revived; note the GPU
//! compression uses a single 128-bit key while `HashState` carries three,
//! so it does not reproduce `hash_two` byte-for-byte as-is.

use std::ffi::CStr;
use std::os::raw::c_char;

#[repr(C)]
struct AESContext {
    _private: [u8; 0],
}

extern "C" {
    fn aes_init() -> *mut AESContext;
    fn aes_set_key(ctx: *mut AESContext, key: *const u32) -> i32;
    fn aes_hash_batch(
        ctx: *mut AESContext,
        h_one: *const u32,
        h_two: *const u32,
        h_out: *mut u32,
        num_elements: i32,
    ) -> i32;
    fn aes_free(ctx: *mut AESContext);
    fn aes_get_last_error() -> *const c_char;
    /// D2D variant: d_one / d_two are device pointers; output stays in ctx->d_out.
    fn aes_hash_batch_d2d(
        ctx: *mut AESContext,
        d_one: *const u32,
        d_two: *const u32,
        num_elements: i32,
    ) -> i32;
    /// Return device pointer to the AES output buffer (ctx->d_out).
    fn aes_get_output_device(ctx: *const AESContext) -> *const u32;
}

/// Rust wrapper for CUDA AES operations.
pub struct CudaAES {
    ctx: *mut AESContext,
}

impl CudaAES {
    /// Initialize a new CUDA AES context (uploads T-tables / S-box once).
    pub fn new() -> Result<Self, String> {
        unsafe {
            let ctx = aes_init();
            if ctx.is_null() {
                return Err(format!(
                    "Failed to initialize AES context: {}",
                    Self::get_cuda_error()
                ));
            }
            Ok(CudaAES { ctx })
        }
    }

    /// Set the AES encryption key (4 × u32 = 128-bit key); runs the key
    /// schedule on device.
    pub fn set_key(&mut self, key: &[u32; 4]) -> Result<(), String> {
        unsafe {
            let result = aes_set_key(self.ctx, key.as_ptr());
            if result != 0 {
                return Err(format!("Failed to set key: {}", Self::get_cuda_error()));
            }
            Ok(())
        }
    }

    /// Batch AES hash: each element is two 256-bit inputs (8 u32s each) →
    /// one 256-bit output.
    pub fn hash_batch(&self, input_one: &[u32], input_two: &[u32]) -> Result<Vec<u32>, String> {
        if input_one.len() != input_two.len() {
            return Err("Input vectors must have the same length".to_string());
        }
        if input_one.len() % 8 != 0 {
            return Err("Input length must be a multiple of 8 (8 u32s per element)".to_string());
        }

        let num_elements = (input_one.len() / 8) as i32;
        let mut output = vec![0u32; input_one.len()];

        unsafe {
            let result = aes_hash_batch(
                self.ctx,
                input_one.as_ptr(),
                input_two.as_ptr(),
                output.as_mut_ptr(),
                num_elements,
            );
            if result != 0 {
                return Err(format!("AES hash batch failed: {}", Self::get_cuda_error()));
            }
        }

        Ok(output)
    }

    /// D2D batch hash: inputs already on device, output stays in the
    /// context's device buffer (retrieve with `device_output_ptr`).
    ///
    /// # Safety
    /// `d_one` / `d_two` must be valid device pointers to
    /// `num_elements * 32` bytes each.
    pub unsafe fn hash_batch_d2d(
        &self,
        d_one: *const u32,
        d_two: *const u32,
        num_elements: usize,
    ) -> Result<(), String> {
        let result = aes_hash_batch_d2d(self.ctx, d_one, d_two, num_elements as i32);
        if result != 0 {
            return Err(format!("AES hash batch d2d failed: {}", Self::get_cuda_error()));
        }
        Ok(())
    }

    /// Raw device pointer to the AES output buffer after the last hash call.
    /// Valid until the next `hash_batch*` or drop.
    pub fn device_output_ptr(&self) -> *const u32 {
        unsafe { aes_get_output_device(self.ctx) }
    }

    /// Raw mutable pointer to the underlying AESContext, cast to `*mut u8`.
    /// Used to hand the AES state to `acss_merkle_batched_d2d`.
    pub fn ctx_mut_ptr(&self) -> *mut u8 {
        self.ctx as *mut u8
    }

    fn get_cuda_error() -> String {
        unsafe {
            let c_str = aes_get_last_error();
            if c_str.is_null() {
                return "unknown CUDA AES error".to_string();
            }
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

impl Drop for CudaAES {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { aes_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

// SAFETY: the context pointer is only ever passed to CUDA runtime calls.
unsafe impl Send for CudaAES {}
unsafe impl Sync for CudaAES {}
