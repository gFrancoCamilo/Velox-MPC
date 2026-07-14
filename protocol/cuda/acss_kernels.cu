// acss_kernels.cu — ACSS-specific GPU kernels for the fused D2D pipeline.
//
// See acss_kernels.cuh for API documentation.
//
// Vendored from async_mpc/secret_sharing/acss_ab/cuda/acss_kernels.cu; the
// per-party DZK evaluation kernel is replaced by the coefficient-space
// acss_compute_dzk_coeffs (Velox broadcasts DZK coefficients instead).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#include "acss_kernels.cuh"
#include "gpu_fields.cuh"

// ---------------------------------------------------------------------------
// Forward declarations of D2D functions from gpu_aes_wrapper.cu. Both objects
// live in the same static archive (libfield_gemm_cuda.a), so the reference is
// resolved intra-archive at host link time.
// ---------------------------------------------------------------------------

extern "C" {
    int aes_hash_batch_d2d(
        void* ctx,
        const uint32_t* d_one,
        const uint32_t* d_two,
        int num_elements
    );
    const uint32_t* aes_get_output_device(const void* ctx);
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

static char last_acss_error[512] = "No error";

static void set_acss_err(const char* msg) {
    snprintf(last_acss_error, sizeof(last_acss_error), "%s", msg);
}

static void set_acss_cuda_err(cudaError_t err) {
    snprintf(last_acss_error, sizeof(last_acss_error), "%s", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Width-generic limb (de)serialisation helpers.
//
// A field element is stored as one or more base-field limbs of `w` bytes each
// (w = 8 for the Mersenne61 family). The on-device / GEMM storage is native
// little-endian per limb; the wire/commitment format is big-endian per limb
// (= to_bytes_be applied to each base-field component of an extension field).
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned long long load_le_bytes_to_u64(const uint8_t* p, int w) {
    unsigned long long v = 0;
    for (int i = 0; i < w; i++) {
        v |= ((unsigned long long)p[i]) << (8 * i);
    }
    return v;
}

__device__ __forceinline__ void store_be_bytes_from_u64(uint8_t* p, unsigned long long v, int w) {
    for (int i = 0; i < w; i++) {
        p[i] = (uint8_t)((v >> (8 * (w - 1 - i))) & 0xFF);
    }
}

__device__ __forceinline__ uint32_t load_be_u32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |
           ((uint32_t)p[3]);
}

__device__ __forceinline__ void store_be_u64(uint8_t* p, unsigned long long v) {
    p[0] = (uint8_t)(v >> 56);
    p[1] = (uint8_t)(v >> 48);
    p[2] = (uint8_t)(v >> 40);
    p[3] = (uint8_t)(v >> 32);
    p[4] = (uint8_t)(v >> 24);
    p[5] = (uint8_t)(v >> 16);
    p[6] = (uint8_t)(v >> 8);
    p[7] = (uint8_t)v;
}

__device__ __forceinline__ void store_fp4_61_be(uint8_t* p, Fp4_61 v) {
    store_be_u64(p,      v.c0.re);
    store_be_u64(p + 8,  v.c0.im);
    store_be_u64(p + 16, v.c1.re);
    store_be_u64(p + 24, v.c1.im);
}

// ---------------------------------------------------------------------------
// Kernel A: Pack per-party share bytes
//
// Grid : (ceil(num_polys / 32), ceil(n_parties / 8))
// Block: (32, 8)
// Each thread handles one (party, poly) pair and copies one share element.
// ---------------------------------------------------------------------------

__global__ void acss_pack_party_shares_kernel(
    const uint8_t* d_rand_cols,
    const uint8_t* d_evals_gemm,
    uint8_t* d_out,
    int t, int n_eval_parties, int num_polys, int party_stride,
    int small_field_bytes, int limb_bytes
) {
    int poly_idx  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int party_idx = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int n_parties = t + n_eval_parties;

    if (poly_idx >= num_polys || party_idx >= n_parties) return;

    // One share is `small_field_bytes` wide.
    // Destination: d_out[party_idx * party_stride + poly_idx * small_field_bytes]
    uint8_t* dst = d_out + (size_t)party_idx * party_stride + (size_t)poly_idx * small_field_bytes;

    const uint8_t* src;
    if (party_idx < t) {
        // PRF-masked share: d_rand_cols[party_idx][poly_idx]
        src = d_rand_cols + ((size_t)party_idx * num_polys + poly_idx) * small_field_bytes;
    } else {
        // Polynomial evaluation: d_evals_gemm[poly_idx][party_idx - t]
        int eval_party = party_idx - t;
        src = d_evals_gemm + ((size_t)poly_idx * n_eval_parties + eval_party) * small_field_bytes;
    }

    // Per-limb native-LE -> big-endian conversion. The source (GEMM/PRF output)
    // is native GPU little-endian per base-field limb; the destination must be
    // big-endian per limb so the layout matches the verifier's
    // `LargeField::to_bytes_be()` reconstruction. Each extension-field
    // component is converted independently.
    int n_limbs = small_field_bytes / limb_bytes;
    for (int limb = 0; limb < n_limbs; limb++) {
        unsigned long long v = load_le_bytes_to_u64(src + (size_t)limb * limb_bytes, limb_bytes);
        store_be_bytes_from_u64(dst + (size_t)limb * limb_bytes, v, limb_bytes);
    }
}

extern "C" int acss_pack_party_shares(
    const uint8_t* d_rand_cols,
    const uint8_t* d_evals_gemm,
    uint8_t* d_out,
    int t, int n_eval_parties, int num_polys, int party_stride,
    int small_field_bytes, int limb_bytes
) {
    dim3 block(32, 8);
    dim3 grid(
        ((unsigned)num_polys  + 31) / 32,
        ((unsigned)(t + n_eval_parties) + 7) / 8
    );
    acss_pack_party_shares_kernel<<<grid, block>>>(
        d_rand_cols, d_evals_gemm, d_out,
        t, n_eval_parties, num_polys, party_stride,
        small_field_bytes, limb_bytes
    );
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel B: Append nonce evaluations
//
// Grid : (ceil(n_parties / 64), 1)
// Block: (64, 1)
// Each thread handles one party; loops over 32 bytes of the nonce.
// ---------------------------------------------------------------------------

__global__ void acss_append_nonce_kernel(
    const uint8_t* d_nonce_evals,
    uint8_t* d_out,
    int n_parties, int party_stride, int nonce_offset
) {
    int party_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (party_idx >= n_parties) return;

    uint8_t* dst = d_out + (size_t)party_idx * party_stride + nonce_offset;
    const uint8_t* src = d_nonce_evals + (size_t)party_idx * 32;

    // The Fp4_61 nonce is 4 × uint64 in native (little-endian on GPU) order.
    // Write each u64 as big-endian to match the verifier's
    // `LargeField::from_bytes_be(...).to_bytes_be()` round-trip, which gives
    // per-u64 BE bytes.
    // `src` is 32-byte aligned (allocated by cudaMalloc), so a u64 read is fine.
    // `dst` is only guaranteed byte aligned, so write byte-by-byte.
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned long long u = ((const unsigned long long*)src)[k];
        uint8_t* d = dst + k * 8;
        d[0] = (uint8_t)((u >> 56) & 0xFF);
        d[1] = (uint8_t)((u >> 48) & 0xFF);
        d[2] = (uint8_t)((u >> 40) & 0xFF);
        d[3] = (uint8_t)((u >> 32) & 0xFF);
        d[4] = (uint8_t)((u >> 24) & 0xFF);
        d[5] = (uint8_t)((u >> 16) & 0xFF);
        d[6] = (uint8_t)((u >>  8) & 0xFF);
        d[7] = (uint8_t)( u        & 0xFF);
    }
}

extern "C" int acss_append_nonce(
    const uint8_t* d_nonce_evals,
    uint8_t* d_out,
    int n_parties, int party_stride, int nonce_offset
) {
    dim3 block(64);
    dim3 grid(((unsigned)n_parties + 63) / 64);
    acss_append_nonce_kernel<<<grid, block>>>(
        d_nonce_evals, d_out, n_parties, party_stride, nonce_offset
    );
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel C: Build blinding commitment AES inputs (dormant — GPU-Merkle path)
// ---------------------------------------------------------------------------

__global__ void acss_build_blinding_hash_inputs_kernel(
    const uint8_t* d_nonce_evals,
    uint32_t* d_one,
    uint32_t* d_two,
    int n_parties
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total_words = n_parties * 8;
    if (idx >= total_words) return;

    int party = idx / 8;
    int word = idx % 8;
    const uint8_t* row1 = d_nonce_evals + ((size_t)n_parties + party) * 32;
    const uint8_t* row2 = d_nonce_evals + ((size_t)2 * n_parties + party) * 32;
    d_one[idx] = load_be_u32(row1 + word * 4);
    d_two[idx] = load_be_u32(row2 + word * 4);
}

extern "C" int acss_build_blinding_hash_inputs(
    const uint8_t* d_nonce_evals,
    uint8_t* d_one,
    uint8_t* d_two,
    int n_parties
) {
    int threads = 256;
    int total_words = n_parties * 8;
    int blocks = (total_words + threads - 1) / threads;
    acss_build_blinding_hash_inputs_kernel<<<blocks, threads>>>(
        d_nonce_evals, (uint32_t*)d_one, (uint32_t*)d_two, n_parties
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel D: DZK coefficient vector on device
//
// dzk[c] = blind_coeffs[c] + Σ_j coeffs[j][c] * powers[j]
//
// Grid : (n_coeffs) — one block per output coefficient (n_coeffs = t+1, small)
// Block: (256)     — threads stride over the num_polys sum dimension
// ---------------------------------------------------------------------------

__global__ void acss_compute_dzk_coeffs_kernel(
    const Fp4_61* d_coeffs,
    const Fp4_61* d_blind_coeffs,
    const Fp4_61* d_powers,
    uint8_t* d_out,
    int num_polys,
    int n_coeffs
) {
    int c = (int)blockIdx.x;
    if (c >= n_coeffs) return;

    __shared__ Fp4_61 partial[256];

    Fp4_61 acc = fp4_61_zero();
    for (int j = (int)threadIdx.x; j < num_polys; j += (int)blockDim.x) {
        acc = fp4_61_add(acc, fp4_61_mul(d_coeffs[(size_t)j * n_coeffs + c], d_powers[j]));
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            partial[threadIdx.x] = fp4_61_add(
                partial[threadIdx.x],
                partial[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Inputs are raw field-GEMM output in native Fp4_61 layout; only the
        // output uses canonical big-endian wire bytes.
        Fp4_61 res = fp4_61_add(d_blind_coeffs[c], partial[0]);
        store_fp4_61_be(d_out + (size_t)c * 32, res);
    }
}

extern "C" int acss_compute_dzk_coeffs(
    const uint8_t* d_coeffs,
    const uint8_t* d_blind_coeffs,
    const uint8_t* d_powers,
    uint8_t* d_out,
    int num_polys,
    int n_coeffs
) {
    dim3 block(256);
    dim3 grid((unsigned)n_coeffs);
    acss_compute_dzk_coeffs_kernel<<<grid, block>>>(
        (const Fp4_61*)d_coeffs,
        (const Fp4_61*)d_blind_coeffs,
        (const Fp4_61*)d_powers,
        d_out,
        num_polys,
        n_coeffs
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Helpers for Merkle batched D2D (dormant — GPU-Merkle path)
// ---------------------------------------------------------------------------

// Extract 32-byte leaves from party data.
// d_leaves[party * n_leaves + leaf] = 32-byte chunk from d_party_data[party][leaf*32..(leaf+1)*32]
//   (zero-padded if beyond party_stride).
// Grid : (n_parties, ceil(n_leaves / 32))
// Block: (32)
__global__ void acss_extract_leaves_kernel(
    const uint8_t* d_party_data,
    uint8_t* d_leaves,          // (n_parties × n_leaves × 32) bytes
    int n_parties, int party_stride, int n_leaves
) {
    int party = (int)blockIdx.x;
    int leaf  = (int)(blockIdx.y * blockDim.x + threadIdx.x);

    if (party >= n_parties || leaf >= n_leaves) return;

    uint8_t* dst = d_leaves + ((size_t)party * n_leaves + leaf) * 32;
    int src_base = leaf * 32;

    #pragma unroll
    for (int b = 0; b < 32; b++) {
        int src_off = src_base + b;
        dst[b] = (src_off < party_stride)
            ? d_party_data[(size_t)party * party_stride + src_off]
            : (uint8_t)0;
    }
}

// Build d_one[party * n_pairs + pair] and d_two[...] from d_nodes.
// Each node is 32 bytes (an AES hash output element).
// Right child of the last pair duplicates the last node (standard Merkle padding).
// Grid : (n_parties, ceil(n_pairs / 32))
// Block: (32)
__global__ void acss_build_pairs_kernel(
    const uint8_t* d_nodes,     // (n_parties × level_len × 32)
    uint8_t* d_one,             // (n_parties × n_pairs × 32)
    uint8_t* d_two,             // (n_parties × n_pairs × 32)
    int n_parties, int level_len, int n_pairs
) {
    int party = (int)blockIdx.x;
    int pair  = (int)(blockIdx.y * blockDim.x + threadIdx.x);

    if (party >= n_parties || pair >= n_pairs) return;

    int left_idx  = pair * 2;
    int right_idx = pair * 2 + 1;
    if (right_idx >= level_len) right_idx = level_len - 1; // pad last

    const uint8_t* left  = d_nodes + ((size_t)party * level_len + left_idx)  * 32;
    const uint8_t* right = d_nodes + ((size_t)party * level_len + right_idx) * 32;

    uint8_t* out_one = d_one + ((size_t)party * n_pairs + pair) * 32;
    uint8_t* out_two = d_two + ((size_t)party * n_pairs + pair) * 32;

    // Copy 32 bytes using 4× 64-bit stores
    ((unsigned long long*)out_one)[0] = ((const unsigned long long*)left)[0];
    ((unsigned long long*)out_one)[1] = ((const unsigned long long*)left)[1];
    ((unsigned long long*)out_one)[2] = ((const unsigned long long*)left)[2];
    ((unsigned long long*)out_one)[3] = ((const unsigned long long*)left)[3];

    ((unsigned long long*)out_two)[0] = ((const unsigned long long*)right)[0];
    ((unsigned long long*)out_two)[1] = ((const unsigned long long*)right)[1];
    ((unsigned long long*)out_two)[2] = ((const unsigned long long*)right)[2];
    ((unsigned long long*)out_two)[3] = ((const unsigned long long*)right)[3];
}

// ---------------------------------------------------------------------------
// Driver: Batched Merkle tree reduction, fully on device (dormant)
//
// Parties are processed in batches sized to fit available device memory,
// so the scratch allocation (d_nodes + d_one_buf + d_two_buf) scales with
// the batch rather than the full n_parties.
// ---------------------------------------------------------------------------

extern "C" const char* acss_get_last_error(void) {
    return last_acss_error;
}

extern "C" int acss_merkle_batched_d2d(
    void* aes_ctx,
    const uint8_t* d_party_data,
    int n_parties,
    int party_stride,
    uint8_t* d_roots_out
) {
    cudaError_t err;

    int n_leaves = (party_stride + 31) / 32;
    if (n_leaves == 0) n_leaves = 1;

    // Special case: single leaf per party — just copy the (zero-padded) leaf.
    if (n_leaves == 1) {
        err = cudaMemcpy2D(
            d_roots_out, 32,
            d_party_data, party_stride,
            (party_stride < 32) ? party_stride : 32, (size_t)n_parties,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
        return 0;
    }

    // Compute safe batch size from available device memory.
    // Memory per party per batch:
    //   d_nodes   : n_leaves × 32
    //   d_one_buf : ceil(n_leaves/2) × 32
    //   d_two_buf : ceil(n_leaves/2) × 32
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Reserve 512 MB for the AES hash output buffer and other overhead.
    size_t usable = (free_mem > (size_t)512 * 1024 * 1024)
                        ? free_mem - (size_t)512 * 1024 * 1024
                        : 0;
    size_t bytes_per_party = (size_t)n_leaves * 32
                           + 2 * (size_t)((n_leaves + 1) / 2) * 32;
    int batch_size = n_parties;
    if (bytes_per_party > 0 && usable / bytes_per_party < (size_t)n_parties) {
        batch_size = (int)(usable / bytes_per_party);
        if (batch_size < 1) batch_size = 1;
    }

    for (int p0 = 0; p0 < n_parties; p0 += batch_size) {
        int this_batch = (p0 + batch_size <= n_parties)
                             ? batch_size
                             : n_parties - p0;

        size_t this_node_bytes = (size_t)this_batch * n_leaves * 32;
        size_t this_pair_bytes = (size_t)this_batch * ((n_leaves + 1) / 2) * 32;

        uint8_t *d_nodes = NULL, *d_one_buf = NULL, *d_two_buf = NULL;
        err = cudaMalloc(&d_nodes,   this_node_bytes);
        if (err != cudaSuccess) { set_acss_cuda_err(err); return -1; }
        err = cudaMalloc(&d_one_buf, this_pair_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_nodes);
            set_acss_cuda_err(err); return -1;
        }
        err = cudaMalloc(&d_two_buf, this_pair_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_nodes); cudaFree(d_one_buf);
            set_acss_cuda_err(err); return -1;
        }

        // Step 1: extract 32-byte leaves from this batch of parties.
        const uint8_t* batch_data = d_party_data + (size_t)p0 * party_stride;
        {
            dim3 block(32);
            dim3 grid((unsigned)this_batch, ((unsigned)n_leaves + 31) / 32);
            acss_extract_leaves_kernel<<<grid, block>>>(
                batch_data, d_nodes, this_batch, party_stride, n_leaves
            );
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                set_acss_cuda_err(err);
                cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
                return -1;
            }
        }

        // Step 2: iterative Merkle reduction for this batch.
        {
            int cur_len = n_leaves;
            while (cur_len > 1) {
                int n_pairs = (cur_len + 1) / 2;
                {
                    dim3 block(32);
                    dim3 grid((unsigned)this_batch, ((unsigned)n_pairs + 31) / 32);
                    acss_build_pairs_kernel<<<grid, block>>>(
                        d_nodes, d_one_buf, d_two_buf,
                        this_batch, cur_len, n_pairs
                    );
                    err = cudaDeviceSynchronize();
                    if (err != cudaSuccess) {
                        set_acss_cuda_err(err);
                        cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
                        return -1;
                    }
                }

                int total_pairs = this_batch * n_pairs;
                int ret = aes_hash_batch_d2d(
                    aes_ctx,
                    (const uint32_t*)d_one_buf,
                    (const uint32_t*)d_two_buf,
                    total_pairs
                );
                if (ret != 0) {
                    set_acss_err("aes_hash_batch_d2d failed in acss_merkle_batched_d2d");
                    cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
                    return -1;
                }

                const uint32_t* d_aes_out = aes_get_output_device(aes_ctx);
                err = cudaMemcpy(d_nodes, d_aes_out,
                                 (size_t)total_pairs * 32,
                                 cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    set_acss_cuda_err(err);
                    cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
                    return -1;
                }

                cur_len = n_pairs;
            }
        }

        // Step 3: copy the roots for this batch to d_roots_out.
        err = cudaMemcpy(d_roots_out + (size_t)p0 * 32, d_nodes,
                         (size_t)this_batch * 32,
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            set_acss_cuda_err(err);
            cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
            return -1;
        }

        cudaFree(d_nodes); cudaFree(d_one_buf); cudaFree(d_two_buf);
    }

    return 0;
}
