//! Smoke test for the CUDA FFI bridge. Only compiled with `--features gpu`;
//! requires an NVIDIA GPU + CUDA runtime present at test time.

#![cfg(feature = "gpu")]

use protocol::gpu_ffi::{CudaFieldGEMM, GpuFieldId};

#[test]
fn ctx_init_and_drop() {
    // Allocates a CUDA context for Mersenne61 Fp4 (32-byte elements), then drops it.
    // Proves the FFI links and the device is reachable. The actual GEMM kernel is
    // exercised by `field_gemm_gpu`.
    let _gemm = CudaFieldGEMM::new(GpuFieldId::M61Fp4, 32)
        .expect("CUDA init failed (no GPU?)");
}
