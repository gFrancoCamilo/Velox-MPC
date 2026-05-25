//! Smoke test for the CUDA FFI bridge. Only compiled with `--features gpu`;
//! requires an NVIDIA GPU + CUDA runtime present at test time.

#![cfg(feature = "gpu")]

use protocol::gpu_ffi::CudaBn254Gemm;

#[test]
fn ctx_init_and_drop() {
    // Allocates a CUDA context, then drops it. Proves the FFI links and the
    // device is reachable. Compute kernel is exercised by Phase C tests.
    let _gemm = CudaBn254Gemm::new().expect("CUDA init failed (no GPU?)");
}
