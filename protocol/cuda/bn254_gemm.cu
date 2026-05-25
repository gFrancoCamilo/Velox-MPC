// SPDX-License-Identifier: Apache-2.0
//
// Phase A stub of the BN254 GEMM C ABI. Real arithmetic + tiled kernel land in
// Phases B and C; this file exists so the Rust FFI can link end-to-end on a
// CUDA-capable box and the smoke test can verify init/free works.

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "bn254_gemm.cuh"

namespace {

struct Bn254GemmContextImpl {
    cudaStream_t stream;
    unsigned char* d_V;
    unsigned char* d_S;
    unsigned char* d_O;
    size_t v_bytes;
    size_t s_bytes;
    size_t o_bytes;
    unsigned int K;
    unsigned int N;
};

constexpr size_t kElemBytes = 32; // 4 × u64

inline bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

} // namespace

extern "C" {

struct Bn254GemmContext* bn254_gemm_init() {
    auto* impl = static_cast<Bn254GemmContextImpl*>(std::calloc(1, sizeof(Bn254GemmContextImpl)));
    if (impl == nullptr) return nullptr;
    if (!cuda_ok(cudaStreamCreate(&impl->stream))) {
        std::free(impl);
        return nullptr;
    }
    return reinterpret_cast<Bn254GemmContext*>(impl);
}

int bn254_gemm_set_matrix(struct Bn254GemmContext* ctx,
                          const unsigned char* v_bytes,
                          unsigned int K,
                          unsigned int N) {
    if (ctx == nullptr || v_bytes == nullptr) return -1;
    auto* impl = reinterpret_cast<Bn254GemmContextImpl*>(ctx);
    const size_t need = static_cast<size_t>(K) * N * kElemBytes;
    if (impl->v_bytes < need) {
        if (impl->d_V != nullptr) cudaFree(impl->d_V);
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&impl->d_V), need))) {
            impl->v_bytes = 0;
            return -2;
        }
        impl->v_bytes = need;
    }
    impl->K = K;
    impl->N = N;
    if (!cuda_ok(cudaMemcpyAsync(impl->d_V, v_bytes, need,
                                 cudaMemcpyHostToDevice, impl->stream))) {
        return -3;
    }
    return cuda_ok(cudaStreamSynchronize(impl->stream)) ? 0 : -4;
}

int bn254_gemm_compute(struct Bn254GemmContext* ctx,
                       const unsigned char* s_bytes,
                       unsigned int batch,
                       unsigned char* o_bytes) {
    // Phase A: not yet implemented. Returning a non-zero status lets the Rust
    // wrapper bubble a clear "GPU GEMM kernel not yet implemented" error.
    (void) ctx;
    (void) s_bytes;
    (void) batch;
    (void) o_bytes;
    return 1; // sentinel: kernel unimplemented in Phase A
}

void bn254_gemm_free(struct Bn254GemmContext* ctx) {
    if (ctx == nullptr) return;
    auto* impl = reinterpret_cast<Bn254GemmContextImpl*>(ctx);
    if (impl->d_V != nullptr) cudaFree(impl->d_V);
    if (impl->d_S != nullptr) cudaFree(impl->d_S);
    if (impl->d_O != nullptr) cudaFree(impl->d_O);
    cudaStreamDestroy(impl->stream);
    std::free(impl);
}

} // extern "C"
