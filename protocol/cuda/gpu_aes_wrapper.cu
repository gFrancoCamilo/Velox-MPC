// gpu_aes_wrapper_complete.cu — optimized wrapper
//
// Three optimizations over the naive version:
//
//  1. PERSISTENT DEVICE BUFFERS
//     AESContext caches d_one/d_two/d_out on the GPU. cudaMalloc is a
//     kernel-mode call (~5-20 µs each). For 1M-element batches called
//     repeatedly that was ~60 µs/call of pure overhead. Now buffers are
//     reallocated only when the batch size grows.
//
//  2. PINNED (PAGE-LOCKED) HOST MEMORY — opt-in via aes_alloc_pinned()
//     cudaMemcpy from normal malloc'd memory must stage through a locked
//     bounce buffer inside the driver. Pinned memory skips that step and
//     roughly doubles PCIe bandwidth (6 GB/s -> 12 GB/s on PCIe 3.0).
//
//  3. CUDA STREAMS + ASYNC TRANSFERS
//     H2D copy, kernel, and D2H copy are all enqueued on one stream
//     and synchronized once at the end. When host memory is pinned the
//     GPU's DMA copy engine can overlap with compute.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "gpu_aes.h"
#include "gpu_aes_kernels.cuh"

// ---------------------------------------------------------------------------
// Inlined from gpu_aes_test.cu (only the two device functions needed)
// ---------------------------------------------------------------------------
__device__ void load_sbox(uint32_t* t0G, uint8_t* SAES_d, uint32_t* rk,
                           uint32_t (*t0S)[32], uint8_t (*Sbox)[32][4], uint32_t* rkS) {
    if (threadIdx.x < TABLE_SIZE) {
        for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
            t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
            Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES_d[threadIdx.x];
        }
        if (threadIdx.x < AES_128_KEY_SIZE_INT) { rkS[threadIdx.x] = rk[threadIdx.x]; }
    }
    __syncthreads();
}

__global__ void aes_key_expansion_kernel(
    u32* d_key, u32* d_expanded, u32* d_rcon,
    u32* t0, u32* t1, u32* t2, u32* t3
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        keyExpansion(d_key, d_expanded, d_rcon, t3, t2, t1, t0);
    }
}

// ---------------------------------------------------------------------------
// Error buffer
// ---------------------------------------------------------------------------
static char last_cuda_error[512] = "No error";
static void set_last_error(cudaError_t err) {
    snprintf(last_cuda_error, sizeof(last_cuda_error), "%s", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// AESContext — persistent GPU buffers + dedicated stream
// ---------------------------------------------------------------------------
// Timing breakdown returned to Rust after each hash_batch call.
// All values are in milliseconds, measured with CUDA events on the GPU timeline.
typedef struct {
    float h2d_ms;     // Host-to-Device transfer (both input buffers)
    float kernel_ms;  // AES kernel execution
    float d2h_ms;     // Device-to-Host transfer (output buffer)
    float total_ms;   // Wall time of the whole pipeline (h2d + kernel + d2h)
} AesTimings;

typedef struct {
    u32* d_t0; u32* d_t1; u32* d_t2; u32* d_t3;
    u32* d_t4; u32* d_t4_0; u32* d_t4_1; u32* d_t4_2; u32* d_t4_3;
    u8*  d_SAES;
    u32* d_rcon;
    u32* d_expanded_key;

    // Persistent I/O buffers
    u32* d_one;
    u32* d_two;
    u32* d_out;
    int  buf_capacity;      // capacity of d_one and d_two (and d_out when grown together)
    int  d_out_capacity;    // capacity of d_out independently (for D2D mode)

    cudaStream_t stream;

    // CUDA events for timing — created once, reused every call
    cudaEvent_t ev_start;      // before H2D
    cudaEvent_t ev_h2d_done;   // after H2D, before kernel
    cudaEvent_t ev_kernel_done;// after kernel, before D2H
    cudaEvent_t ev_end;        // after D2H

    bool initialized;
} AESContext;

extern "C" {

AESContext* aes_init() {
    AESContext* ctx = (AESContext*)calloc(1, sizeof(AESContext));
    if (!ctx) return NULL;

    cudaError_t err;

#define ALLOC_TABLE(field, size) \
    err = cudaMalloc(&ctx->field, (size) * sizeof(u32)); \
    if (err != cudaSuccess) { set_last_error(err); goto cleanup; }

    ALLOC_TABLE(d_t0,   TABLE_SIZE)
    ALLOC_TABLE(d_t1,   TABLE_SIZE)
    ALLOC_TABLE(d_t2,   TABLE_SIZE)
    ALLOC_TABLE(d_t3,   TABLE_SIZE)
    ALLOC_TABLE(d_t4,   TABLE_SIZE)
    ALLOC_TABLE(d_t4_0, TABLE_SIZE)
    ALLOC_TABLE(d_t4_1, TABLE_SIZE)
    ALLOC_TABLE(d_t4_2, TABLE_SIZE)
    ALLOC_TABLE(d_t4_3, TABLE_SIZE)
#undef ALLOC_TABLE

    err = cudaMalloc(&ctx->d_SAES, 256 * sizeof(u8));
    if (err != cudaSuccess) { set_last_error(err); goto cleanup; }
    err = cudaMalloc(&ctx->d_rcon, RCON_SIZE * sizeof(u32));
    if (err != cudaSuccess) { set_last_error(err); goto cleanup; }
    err = cudaMalloc(&ctx->d_expanded_key, 44 * sizeof(u32));
    if (err != cudaSuccess) { set_last_error(err); goto cleanup; }

    cudaMemcpy(ctx->d_t0,   T0,    TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t1,   T1,    TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t2,   T2,    TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t3,   T3,    TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t4,   T4,    TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t4_0, T4_0,  TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t4_1, T4_1,  TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t4_2, T4_2,  TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_t4_3, T4_3,  TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_SAES, SAES,  256        * sizeof(u8),  cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_rcon, RCON32,RCON_SIZE  * sizeof(u32), cudaMemcpyHostToDevice);

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) { set_last_error(err); goto cleanup; }

    // Create timing events once — reused on every hash_batch call
    cudaEventCreate(&ctx->ev_start);
    cudaEventCreate(&ctx->ev_h2d_done);
    cudaEventCreate(&ctx->ev_kernel_done);
    cudaEventCreate(&ctx->ev_end);

    ctx->d_one = ctx->d_two = ctx->d_out = NULL;
    ctx->buf_capacity = 0;
    ctx->d_out_capacity = 0;
    ctx->initialized = true;
    return ctx;

cleanup:
    if (ctx->d_t0)   cudaFree(ctx->d_t0);
    if (ctx->d_t1)   cudaFree(ctx->d_t1);
    if (ctx->d_t2)   cudaFree(ctx->d_t2);
    if (ctx->d_t3)   cudaFree(ctx->d_t3);
    if (ctx->d_t4)   cudaFree(ctx->d_t4);
    if (ctx->d_t4_0) cudaFree(ctx->d_t4_0);
    if (ctx->d_t4_1) cudaFree(ctx->d_t4_1);
    if (ctx->d_t4_2) cudaFree(ctx->d_t4_2);
    if (ctx->d_t4_3) cudaFree(ctx->d_t4_3);
    if (ctx->d_SAES) cudaFree(ctx->d_SAES);
    if (ctx->d_rcon) cudaFree(ctx->d_rcon);
    if (ctx->d_expanded_key) cudaFree(ctx->d_expanded_key);
    free(ctx);
    return NULL;
}

int aes_set_key(AESContext* ctx, const uint32_t* key) {
    if (!ctx || !ctx->initialized) return -1;

    u32* d_initial_key;
    cudaError_t err = cudaMalloc(&d_initial_key, 4 * sizeof(u32));
    if (err != cudaSuccess) { set_last_error(err); return -1; }

    cudaMemcpy(d_initial_key, key, 4 * sizeof(u32), cudaMemcpyHostToDevice);
    aes_key_expansion_kernel<<<1, 1>>>(
        d_initial_key, ctx->d_expanded_key, ctx->d_rcon,
        ctx->d_t4_0, ctx->d_t4_1, ctx->d_t4_2, ctx->d_t4_3
    );
    cudaError_t key_err = cudaDeviceSynchronize();
    cudaFree(d_initial_key);
    if (key_err != cudaSuccess) {
        set_last_error(key_err);
        fprintf(stderr, "[cuda-aes] Key expansion failed: %s\n", cudaGetErrorString(key_err));
        return -1;
    }
    return 0;
}

// Grow all three persistent I/O buffers only when capacity is exceeded.
// Used by the host-input path (aes_hash_batch / aes_hash_batch_timed).
static int ensure_capacity(AESContext* ctx, int num_elements) {
    if (num_elements <= ctx->buf_capacity) return 0;

    if (ctx->d_one) cudaFree(ctx->d_one);
    if (ctx->d_two) cudaFree(ctx->d_two);
    if (ctx->d_out) cudaFree(ctx->d_out);
    ctx->d_one = ctx->d_two = ctx->d_out = NULL;
    ctx->buf_capacity = 0;
    ctx->d_out_capacity = 0;

    size_t bytes = (size_t)num_elements * 8 * sizeof(u32);
    cudaError_t err;
    err = cudaMalloc(&ctx->d_one, bytes); if (err != cudaSuccess) { set_last_error(err); return -1; }
    err = cudaMalloc(&ctx->d_two, bytes); if (err != cudaSuccess) { set_last_error(err); return -1; }
    err = cudaMalloc(&ctx->d_out, bytes); if (err != cudaSuccess) { set_last_error(err); return -1; }

    ctx->buf_capacity = num_elements;
    ctx->d_out_capacity = num_elements;
    return 0;
}


int aes_hash_batch_timed(
    AESContext*     ctx,
    const uint32_t* h_one,
    const uint32_t* h_two,
    uint32_t*       h_out,
    int             num_elements,
    AesTimings*     timings_out   // may be NULL
) {
    if (!ctx || !ctx->initialized) return -1;
    if (ensure_capacity(ctx, num_elements) != 0) return -1;

    size_t bytes = (size_t)num_elements * 8 * sizeof(u32);

    // --- Record: start of H2D ---
    cudaEventRecord(ctx->ev_start, ctx->stream);

    cudaMemcpyAsync(ctx->d_one, h_one, bytes, cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(ctx->d_two, h_two, bytes, cudaMemcpyHostToDevice, ctx->stream);

    // --- Record: H2D done, kernel starting ---
    cudaEventRecord(ctx->ev_h2d_done, ctx->stream);

    cudaFuncSetCacheConfig(aes_hash_batch_kernel, cudaFuncCachePreferShared);
    int threads = 256;
    int blocks  = (num_elements + threads - 1) / threads;
    aes_hash_batch_kernel<<<blocks, threads, 0, ctx->stream>>>(
        ctx->d_one, ctx->d_two, ctx->d_out,
        ctx->d_expanded_key, ctx->d_t0, ctx->d_SAES,
        num_elements
    );

    // --- Record: kernel done, D2H starting ---
    cudaEventRecord(ctx->ev_kernel_done, ctx->stream);

    cudaMemcpyAsync(h_out, ctx->d_out, bytes, cudaMemcpyDeviceToHost, ctx->stream);

    // --- Record: D2H done ---
    cudaEventRecord(ctx->ev_end, ctx->stream);

    // Single sync — waits for all enqueued work to finish
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        set_last_error(err);
        fprintf(stderr, "[cuda-aes] hash_batch stream failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(err);
        fprintf(stderr, "[cuda-aes] hash_batch kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Fill timing breakdown if requested.
    // cudaEventElapsed measures GPU-side elapsed time between two recorded events,
    // so it correctly captures async overlap that CPU timers would miss.
    if (timings_out) {
        cudaEventElapsedTime(&timings_out->h2d_ms,    ctx->ev_start,      ctx->ev_h2d_done);
        cudaEventElapsedTime(&timings_out->kernel_ms, ctx->ev_h2d_done,   ctx->ev_kernel_done);
        cudaEventElapsedTime(&timings_out->d2h_ms,    ctx->ev_kernel_done,ctx->ev_end);
        cudaEventElapsedTime(&timings_out->total_ms,  ctx->ev_start,      ctx->ev_end);
    }

    return 0;
}

// aes_hash_batch — runs the pipeline and optionally fills *timings_out.
// Pass NULL for timings_out if you don't need the breakdown.
int aes_hash_batch(
    AESContext*     ctx,
    const uint32_t* h_one,
    const uint32_t* h_two,
    uint32_t*       h_out,
    int             num_elements
) {
    return aes_hash_batch_timed(ctx, h_one, h_two, h_out, num_elements, NULL);
}

// ---------------------------------------------------------------------------
// Device-to-device variants: both inputs and output stay on the GPU.
// Use these to chain AES hashing with upstream GPU operations (e.g. GEMM)
// without an intermediate D2H + H2D round-trip.
// ---------------------------------------------------------------------------

/// Like aes_hash_batch but d_one / d_two are device pointers (no H2D);
/// output stays in ctx->d_out (no D2H). Retrieve via aes_get_output_device().
int aes_hash_batch_d2d(
    AESContext*     ctx,
    const uint32_t* d_one,   // device pointer — 8*num_elements u32s (caller-owned)
    const uint32_t* d_two,   // device pointer — 8*num_elements u32s (caller-owned)
    int             num_elements
) {
    if (!ctx || !ctx->initialized) return -1;
    // Inputs are caller-supplied device pointers — only ctx->d_out is needed.
    // Avoid ensure_capacity() which would wastefully allocate ctx->d_one and
    // ctx->d_two (8× num_elements × 32 bytes that are never used here).
    if (num_elements > ctx->d_out_capacity) {
        if (ctx->d_out) { cudaFree(ctx->d_out); ctx->d_out = NULL; ctx->d_out_capacity = 0; }
        size_t bytes = (size_t)num_elements * 8 * sizeof(u32);
        cudaError_t err = cudaMalloc(&ctx->d_out, bytes);
        if (err != cudaSuccess) { set_last_error(err); return -1; }
        ctx->d_out_capacity = num_elements;
    }

    cudaFuncSetCacheConfig(aes_hash_batch_kernel, cudaFuncCachePreferShared);
    int threads = 256;
    int blocks  = (num_elements + threads - 1) / threads;
    aes_hash_batch_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (u32*)d_one, (u32*)d_two, ctx->d_out,
        ctx->d_expanded_key, ctx->d_t0, ctx->d_SAES,
        num_elements
    );

    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        set_last_error(err);
        fprintf(stderr, "[cuda-aes-d2d] stream sync: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(err);
        fprintf(stderr, "[cuda-aes-d2d] kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/// Return the raw device pointer to the AES output buffer (ctx->d_out).
/// Valid until the next call to aes_hash_batch* or aes_free on this context.
const uint32_t* aes_get_output_device(const AESContext* ctx) {
    if (!ctx) return NULL;
    return ctx->d_out;
}

// ---------------------------------------------------------------------------
// Pinned memory helpers — use these for h_one/h_two/h_out buffers that are
// reused across multiple hash_batch calls to get full PCIe bandwidth.
// ---------------------------------------------------------------------------
uint32_t* aes_alloc_pinned(int num_u32s) {
    void* ptr = NULL;
    cudaError_t err = cudaMallocHost(&ptr, (size_t)num_u32s * sizeof(u32));
    if (err != cudaSuccess) { set_last_error(err); return NULL; }
    return (uint32_t*)ptr;
}

void aes_free_pinned(uint32_t* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// ---------------------------------------------------------------------------
// aes_hash_merkle (unchanged logic, uses ctx->stream)
// ---------------------------------------------------------------------------
int aes_hash_merkle(
    AESContext*    ctx,
    const uint8_t* h_input,
    int            input_len,
    uint32_t*      h_out
) {
    if (!ctx || !ctx->initialized) return -1;

    int num_leaves = (input_len + 31) / 32;
    if (num_leaves == 0) num_leaves = 1;

    size_t buf_bytes = (size_t)num_leaves * 32;
    u32 *d_cur = nullptr, *d_next = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_cur,  buf_bytes); if (err != cudaSuccess) { set_last_error(err); return -1; }
    err = cudaMalloc(&d_next, buf_bytes); if (err != cudaSuccess) { set_last_error(err); cudaFree(d_cur); return -1; }

    cudaMemset(d_cur, 0, buf_bytes);
    if (input_len > 0) {
        err = cudaMemcpy(d_cur, h_input, input_len, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { set_last_error(err); goto merkle_cleanup; }
    }

    {
        cudaFuncSetCacheConfig(merkle_hash_level_kernel, cudaFuncCachePreferShared);
        int cur_nodes = num_leaves;
        while (cur_nodes > 1) {
            int num_pairs = (cur_nodes + 1) / 2;
            int blocks    = (num_pairs + 255) / 256;
            merkle_hash_level_kernel<<<blocks, 256, 0, ctx->stream>>>(
                d_cur, d_next, ctx->d_expanded_key, ctx->d_t0, ctx->d_SAES, cur_nodes
            );
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) { set_last_error(err); goto merkle_cleanup; }
            u32* tmp = d_cur; d_cur = d_next; d_next = tmp;
            cur_nodes = num_pairs;
        }
        err = cudaMemcpy(h_out, d_cur, 32, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { set_last_error(err); goto merkle_cleanup; }
    }

    cudaFree(d_cur); cudaFree(d_next);
    return 0;

merkle_cleanup:
    cudaFree(d_cur); cudaFree(d_next);
    return -1;
}

void aes_free(AESContext* ctx) {
    if (!ctx) return;
    if (ctx->initialized) {
        cudaFree(ctx->d_t0);   cudaFree(ctx->d_t1);
        cudaFree(ctx->d_t2);   cudaFree(ctx->d_t3);
        cudaFree(ctx->d_t4);   cudaFree(ctx->d_t4_0);
        cudaFree(ctx->d_t4_1); cudaFree(ctx->d_t4_2);
        cudaFree(ctx->d_t4_3); cudaFree(ctx->d_SAES);
        cudaFree(ctx->d_rcon); cudaFree(ctx->d_expanded_key);
        if (ctx->d_one) cudaFree(ctx->d_one);
        if (ctx->d_two) cudaFree(ctx->d_two);
        if (ctx->d_out) cudaFree(ctx->d_out);
        cudaEventDestroy(ctx->ev_start);
        cudaEventDestroy(ctx->ev_h2d_done);
        cudaEventDestroy(ctx->ev_kernel_done);
        cudaEventDestroy(ctx->ev_end);
        cudaStreamDestroy(ctx->stream);
    }
    free(ctx);
}

const char* aes_get_last_error() { return last_cuda_error; }

} // extern "C"

