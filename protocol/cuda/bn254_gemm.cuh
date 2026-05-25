// SPDX-License-Identifier: Apache-2.0
//
// C ABI for the BN254 matrix-matrix multiply kernel.
//
// Memory layout contract:
//   * Every field element is a bn254_fp_t — 4 little-endian u64 limbs (32 B)
//     in lambdaworks Montgomery form. Byte-identical to Rust
//     `FieldElement<BN254PrimeField>`.
//   * V is a row-major K×N matrix (K rows, N cols) of bn254_fp_t.
//   * S is a row-major batch×N matrix (batch rows, N cols) of bn254_fp_t.
//   * O is a row-major batch×K matrix produced such that
//        O[i][r] = Σ_l V[r][l] * S[i][l]
//     i.e. the "matrix dot vectors" layout used by `matrix_matrix_multiply`.
//
// Phase A (this header lands first): symbols exist with empty bodies so the
// Rust FFI can link. Phases B–C fill them in with real arithmetic.

#ifndef BN254_GEMM_CUH
#define BN254_GEMM_CUH

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context owning device buffers (d_V, d_S, d_O) + a CUDA stream.
// The Rust side never inspects the layout — it only holds the pointer.
struct Bn254GemmContext;

// Allocate the context and initialise the CUDA device. Returns NULL on failure
// (no device, alloc failure, …).
struct Bn254GemmContext* bn254_gemm_init();

// Upload a K×N V matrix. v_bytes must point to K*N*32 contiguous bytes.
// Returns 0 on success, non-zero on failure.
int bn254_gemm_set_matrix(struct Bn254GemmContext* ctx,
                          const unsigned char* v_bytes,
                          unsigned int K,
                          unsigned int N);

// Run the GEMM. s_bytes is batch*N*32 bytes; o_bytes receives batch*K*32 bytes.
// Returns 0 on success, non-zero on failure.
int bn254_gemm_compute(struct Bn254GemmContext* ctx,
                       const unsigned char* s_bytes,
                       unsigned int batch,
                       unsigned char* o_bytes);

// Release device buffers + context.
void bn254_gemm_free(struct Bn254GemmContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // BN254_GEMM_CUH
