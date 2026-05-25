// gpu_field_gemm.cuh — Templated tiled GEMM kernel over finite field extensions.
//
// O = S × V   where S is (batch × K), V is (K × N), O is (batch × N).
//
// All arithmetic goes through FieldOps<F>::add / mul / zero so the same
// kernel source serves every field type (M61 base, Fp2_61, Fp4_61) declared
// in gpu_fields.cuh.
//
// Ported verbatim from async_mpc/fields/cuda/gpu_field_gemm.cuh.

#pragma once
#include "gpu_fields.cuh"

#ifndef FIELD_TILE
#define FIELD_TILE 16
#endif

template<typename F>
__global__ void fieldGEMM(const F* __restrict__ S,
                          const F* __restrict__ V,
                          F*       __restrict__ O,
                          int batch, int K, int N)
{
    int row = blockIdx.y * FIELD_TILE + threadIdx.y;  // batch index
    int col = blockIdx.x * FIELD_TILE + threadIdx.x;  // N index

    F acc = FieldOps<F>::zero();

    for (int t = 0; t < (K + FIELD_TILE - 1) / FIELD_TILE; ++t) {
        __shared__ F tS[FIELD_TILE][FIELD_TILE];
        __shared__ F tV[FIELD_TILE][FIELD_TILE];

        int sk = t * FIELD_TILE + threadIdx.x;
        tS[threadIdx.y][threadIdx.x] =
            (row < batch && sk < K) ? S[row * K + sk] : FieldOps<F>::zero();

        int vk = t * FIELD_TILE + threadIdx.y;
        tV[threadIdx.y][threadIdx.x] =
            (vk < K && col < N) ? V[vk * N + col] : FieldOps<F>::zero();

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < FIELD_TILE; ++k) {
            acc = FieldOps<F>::add(acc, FieldOps<F>::mul(tS[threadIdx.y][k],
                                                         tV[k][threadIdx.x]));
        }

        __syncthreads();
    }

    if (row < batch && col < N)
        O[row * N + col] = acc;
}
