// SPDX-License-Identifier: Apache-2.0
//
// C ABI wrapper for the Mersenne-61 family GEMM kernel.
//
// Memory layout contract:
//   - Element bytes vary by field (M61_FP=8, M61_FP2=16, M61_FP4=32). Caller
//     passes a `field_id` to `field_gemm_set_matrix` and we look up the per-
//     element width via `elem_size_for`.
//   - V is row-major (K × N) of `elem_size` bytes per element.
//   - S is row-major (batch × K) of `elem_size` bytes per element.
//   - O is row-major (batch × N) of `elem_size` bytes per element.
//
// Compute dispatches on `field_id` to the appropriate `fieldGEMM<F>` template
// instantiation (M61 base = u64, Fp2_61, Fp4_61).
//
// Adapted from async_mpc/fields/cuda/gpu_field_gemm_wrapper.cu (M31 dispatch
// removed; velox's protocol field is Mersenne61 Fp4).

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "gpu_field_gemm.cuh"

// ---- Field identification — kept binary-compatible with async_mpc ----

enum FieldId {
    FIELD_M61_FP   = 4,
    FIELD_M61_FP2  = 5,
    FIELD_M61_FP4  = 6,
};

static size_t elem_size_for(int field_id) {
    switch (field_id) {
        case FIELD_M61_FP:  return sizeof(unsigned long long);   // 8
        case FIELD_M61_FP2: return sizeof(Fp2_61);                // 16
        case FIELD_M61_FP4: return sizeof(Fp4_61);                // 32
        default:            return 0;
    }
}

// ---- Internal context ----

namespace {

struct FieldGEMMContextImpl {
    cudaStream_t   stream;
    unsigned char* d_V;
    unsigned char* d_S;
    unsigned char* d_O;
    size_t         v_capacity;   // current d_V allocation in bytes
    size_t         s_capacity;
    size_t         o_capacity;
    unsigned int   K;
    unsigned int   N;
    int            field_id;     // set by field_gemm_set_matrix
};

inline bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

// Free any prior allocation, then cudaMalloc `need` bytes into *out.
static int ensure_capacity(unsigned char** out, size_t* capacity, size_t need) {
    if (*capacity >= need) return 0;
    if (*out != nullptr) cudaFree(*out);
    *out = nullptr;
    *capacity = 0;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(out), need))) {
        return -1;
    }
    *capacity = need;
    return 0;
}

template<typename F>
static int launch_gemm_typed(const FieldGEMMContextImpl* impl,
                             unsigned int batch) {
    const F* d_S = reinterpret_cast<const F*>(impl->d_S);
    const F* d_V = reinterpret_cast<const F*>(impl->d_V);
    F*       d_O = reinterpret_cast<F*>(impl->d_O);

    dim3 block(FIELD_TILE, FIELD_TILE);
    dim3 grid((impl->N + FIELD_TILE - 1) / FIELD_TILE,
              (batch    + FIELD_TILE - 1) / FIELD_TILE);

    fieldGEMM<F><<<grid, block, 0, impl->stream>>>(
        d_S, d_V, d_O,
        static_cast<int>(batch),
        static_cast<int>(impl->K),
        static_cast<int>(impl->N));
    return cuda_ok(cudaGetLastError()) ? 0 : -1;
}

} // namespace

// ---- C ABI ----

extern "C" {

struct FieldGEMMContext;  // opaque to callers

struct FieldGEMMContext* field_gemm_init() {
    auto* impl = static_cast<FieldGEMMContextImpl*>(std::calloc(1, sizeof(FieldGEMMContextImpl)));
    if (impl == nullptr) return nullptr;
    if (!cuda_ok(cudaStreamCreate(&impl->stream))) {
        std::free(impl);
        return nullptr;
    }
    return reinterpret_cast<FieldGEMMContext*>(impl);
}

// Upload a K × N V matrix of elements identified by `field_id`.
int field_gemm_set_matrix(struct FieldGEMMContext* ctx,
                          const unsigned char* v_bytes,
                          unsigned int K,
                          unsigned int N,
                          int field_id) {
    if (ctx == nullptr || v_bytes == nullptr) return -1;
    auto* impl = reinterpret_cast<FieldGEMMContextImpl*>(ctx);
    const size_t es = elem_size_for(field_id);
    if (es == 0) return -2;

    const size_t need = static_cast<size_t>(K) * N * es;
    if (ensure_capacity(&impl->d_V, &impl->v_capacity, need) != 0) return -3;

    if (!cuda_ok(cudaMemcpyAsync(impl->d_V, v_bytes, need,
                                 cudaMemcpyHostToDevice, impl->stream))) {
        return -4;
    }
    if (!cuda_ok(cudaStreamSynchronize(impl->stream))) return -5;

    impl->K        = K;
    impl->N        = N;
    impl->field_id = field_id;
    return 0;
}

// Run the GEMM; on return o_bytes contains batch × N elements of the field
// matching the previously-set field_id.
int field_gemm_compute(struct FieldGEMMContext* ctx,
                       const unsigned char* s_bytes,
                       unsigned int batch,
                       unsigned char* o_bytes) {
    if (ctx == nullptr || s_bytes == nullptr || o_bytes == nullptr) return -1;
    auto* impl = reinterpret_cast<FieldGEMMContextImpl*>(ctx);

    const size_t es = elem_size_for(impl->field_id);
    if (es == 0) return -2;

    const size_t s_need = static_cast<size_t>(batch) * impl->K * es;
    const size_t o_need = static_cast<size_t>(batch) * impl->N * es;
    if (ensure_capacity(&impl->d_S, &impl->s_capacity, s_need) != 0) return -3;
    if (ensure_capacity(&impl->d_O, &impl->o_capacity, o_need) != 0) return -4;

    if (!cuda_ok(cudaMemcpyAsync(impl->d_S, s_bytes, s_need,
                                 cudaMemcpyHostToDevice, impl->stream))) {
        return -5;
    }

    int rc = 0;
    switch (impl->field_id) {
        case FIELD_M61_FP:  rc = launch_gemm_typed<unsigned long long>(impl, batch); break;
        case FIELD_M61_FP2: rc = launch_gemm_typed<Fp2_61>(impl, batch); break;
        case FIELD_M61_FP4: rc = launch_gemm_typed<Fp4_61>(impl, batch); break;
        default: rc = -6;
    }
    if (rc != 0) return rc;

    if (!cuda_ok(cudaMemcpyAsync(o_bytes, impl->d_O, o_need,
                                 cudaMemcpyDeviceToHost, impl->stream))) {
        return -7;
    }
    return cuda_ok(cudaStreamSynchronize(impl->stream)) ? 0 : -8;
}

void field_gemm_free(struct FieldGEMMContext* ctx) {
    if (ctx == nullptr) return;
    auto* impl = reinterpret_cast<FieldGEMMContextImpl*>(ctx);
    if (impl->d_V != nullptr) cudaFree(impl->d_V);
    if (impl->d_S != nullptr) cudaFree(impl->d_S);
    if (impl->d_O != nullptr) cudaFree(impl->d_O);
    cudaStreamDestroy(impl->stream);
    std::free(impl);
}

} // extern "C"
