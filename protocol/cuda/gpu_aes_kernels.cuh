//#include "gpu_aes.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>

#include <cuda_runtime.h>


#include <device_launch_parameters.h>
//#include <device_functions.h>

#ifndef GET_BYTE
#define GET_BYTE(x, n) (((x) >> (8 * (n))) & 0xFF)
#endif
#ifndef ROT_L
#define ROT_L(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#endif

__device__ u32 arithmeticRightShift(u32 x, u32 n) { return (x >> n) | (x << (-n & 31)); }
__device__ u32 arithmetic16bitRightShift(u32 x, u32 n, u32 n2Power) { return (x >> n) | ((x & n2Power) << (-n & 15)); }
__device__ u32 arithmeticRightShiftBytePerm(u32 x, u32 n) { return __byte_perm(x, x, n); }

// Wrapping addition of 4 bytes packed in u32
__device__ __forceinline__ u32 d_add_bytes(u32 a, u32 b) {
    u32 res = 0;
    res |= ((GET_BYTE(a, 0) + GET_BYTE(b, 0)) & 0xFF);
    res |= ((GET_BYTE(a, 1) + GET_BYTE(b, 1)) & 0xFF) << 8;
    res |= ((GET_BYTE(a, 2) + GET_BYTE(b, 2)) & 0xFF) << 16;
    res |= ((GET_BYTE(a, 3) + GET_BYTE(b, 3)) & 0xFF) << 24;
    return res;
}

// Multiplication by 2 over GF(2^8) wrapping
__device__ __forceinline__ u32 d_mul2_bytes(u32 a) {
    u32 res = 0;
    res |= ((GET_BYTE(a, 0) << 1) & 0xFF);
    res |= ((GET_BYTE(a, 1) << 1) & 0xFF) << 8;
    res |= ((GET_BYTE(a, 2) << 1) & 0xFF) << 16;
    res |= ((GET_BYTE(a, 3) << 1) & 0xFF) << 24;
    return res;
}

// Key expansion and AES implementation below based on Cihangir Tezcan's implementation
__device__ void keyExpansion(u32* key, u32* rk, u32* rcon, u32* t4_3, u32* t4_2, u32* t4_1, u32* t4_0) {

	//u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	u32 rk0, rk1, rk2, rk3;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
		u32 temp = rk3;
		rk0 = rk0 ^ t4_3[(temp >> 16) & 0xff] ^ t4_2[(temp >> 8) & 0xff] ^ t4_1[(temp) & 0xff] ^ t4_0[(temp >> 24)] ^ rcon[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;

		rk[roundCount * 4 + 4] = rk0;
		rk[roundCount * 4 + 5] = rk1;
		rk[roundCount * 4 + 6] = rk2;
		rk[roundCount * 4 + 7] = rk3;
	}
}

__device__ void aes_encrypt (u32* pt, u32* rkS, u32 (*t0S)[32], u8 (*Sbox)[32][4], int warpThreadIndex) {
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];
	s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
	s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3];
	u32 t0, t1, t2, t3;
	for (u8 roundCount = 0; roundCount < 9; roundCount++) {
		u32 rkStart = roundCount * 4 + 4;
		t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
		t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
		t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
		t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];
		s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
	}
	s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rkS[40];
	s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rkS[41];
	s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rkS[42];
	s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rkS[43];
	pt[0] = s0;
	pt[1] = s1;
	pt[2] = s2;
	pt[3] = s3;
}

// Helper: byte-swap 4 consecutive u32s (LE <-> BE for AES T-table compatibility)
// aes_encrypt expects big-endian u32s; our data is little-endian.
// Every AES call must be wrapped: bswap4 before, bswap4 after.
#define BSWAP4(arr) do { \
    (arr)[0] = __byte_perm((arr)[0], 0, 0x0123); \
    (arr)[1] = __byte_perm((arr)[1], 0, 0x0123); \
    (arr)[2] = __byte_perm((arr)[2], 0, 0x0123); \
    (arr)[3] = __byte_perm((arr)[3], 0, 0x0123); \
} while(0)

__global__ void aes_hash_batch_kernel(
    u32* one,
    u32* two,
    u32* out,
    u32* rk_all,
    u32* t0G,
    u8* SAES_d,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x & 31;

    __shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
    __shared__ u8 Sbox[64][32][4];

    if (threadIdx.x < TABLE_SIZE) {
        for (u8 bank = 0; bank < SHARED_MEM_BANK_SIZE; bank++) {
            t0S[threadIdx.x][bank] = t0G[threadIdx.x];
            Sbox[threadIdx.x / 4][bank][threadIdx.x % 4] = SAES_d[threadIdx.x];
        }
    }
    __syncthreads();

    if (idx >= num_elements) return;

    const u32* o_ptr = one + (idx * 8);
    const u32* t_ptr = two + (idx * 8);

    u32 O[8], T[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { O[i] = o_ptr[i]; T[i] = t_ptr[i]; }

    // --- Layer 0: blk = AES_k0( O[0:4] + 2*T[0:4] ) || AES_k0( O[4:8] + 2*T[4:8] ) ---
    u32 L0[8];
    u32* L0_1 = &L0[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L0[i] = d_add_bytes(O[i], d_mul2_bytes(T[i]));
    BSWAP4(L0);    aes_encrypt(L0,   rk_all, t0S, Sbox, warpIdx);  BSWAP4(L0);
    BSWAP4(L0_1);  aes_encrypt(L0_1, rk_all, t0S, Sbox, warpIdx);  BSWAP4(L0_1);

    // --- Layer 1: blk = AES_k1( 2*O + 2*T + L0[0:4] ) || AES_k1( 2*O + 2*T + L0[4:8] ) ---
    u32 L1[8];
    u32* L1_1 = &L1[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L1[i] = d_add_bytes(d_add_bytes(d_mul2_bytes(O[i]), d_mul2_bytes(T[i])), L0[i]);
    BSWAP4(L1);    aes_encrypt(L1,   rk_all, t0S, Sbox, warpIdx);  BSWAP4(L1);
    BSWAP4(L1_1);  aes_encrypt(L1_1, rk_all, t0S, Sbox, warpIdx);  BSWAP4(L1_1);

    // --- Layer 2: blk = AES_k2( 2*O + T + L1[0:4] ) || AES_k2( 2*O + T + L1[4:8] ) ---
    u32 L2[8];
    u32* L2_1 = &L2[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L2[i] = d_add_bytes(d_add_bytes(d_mul2_bytes(O[i]), T[i]), L1[i]);
    BSWAP4(L2);    aes_encrypt(L2,   rk_all, t0S, Sbox, warpIdx);  BSWAP4(L2);
    BSWAP4(L2_1);  aes_encrypt(L2_1, rk_all, t0S, Sbox, warpIdx);  BSWAP4(L2_1);

    // --- Final: out = O + L0 + L1 + 2*L2 (all in LE, no final byte-swap needed) ---
    u32* res_ptr = out + (idx * 8);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        res_ptr[i] = d_add_bytes(d_add_bytes(d_add_bytes(O[i], L0[i]), L1[i]), d_mul2_bytes(L2[i]));
    }
}


// ---------------------------------------------------------------------------
// Merkle tree level kernel — same hash logic as aes_hash_batch_kernel
// but reads pairs from a single flat node array.
// ---------------------------------------------------------------------------
__global__ void merkle_hash_level_kernel(
    const u32* __restrict__ nodes,
    u32*       __restrict__ out,
    const u32* __restrict__ rk_all,
    const u32* __restrict__ t0G,
    const u8*  __restrict__ SAES_d,
    int num_nodes
) {
    int pair_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = (num_nodes + 1) / 2;
    if (pair_idx >= num_pairs) return;

    int warpIdx = threadIdx.x & 31;

    __shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
    __shared__ u8  Sbox[64][32][4];

    if (threadIdx.x < TABLE_SIZE) {
        for (u8 bank = 0; bank < SHARED_MEM_BANK_SIZE; bank++) {
            t0S[threadIdx.x][bank] = t0G[threadIdx.x];
            Sbox[threadIdx.x / 4][bank][threadIdx.x % 4] = SAES_d[threadIdx.x];
        }
    }
    __syncthreads();

    int left_idx  = pair_idx * 2;
    int right_idx = (left_idx + 1 < num_nodes) ? left_idx + 1 : left_idx;

    const u32* L = nodes + left_idx  * 8;
    const u32* R = nodes + right_idx * 8;

    u32 O[8], T[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { O[i] = L[i]; T[i] = R[i]; }

    // --- Layer 0 ---
    u32 L0[8]; u32* L0_1 = &L0[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L0[i] = d_add_bytes(O[i], d_mul2_bytes(T[i]));
    BSWAP4(L0);    aes_encrypt(L0,   (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L0);
    BSWAP4(L0_1);  aes_encrypt(L0_1, (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L0_1);

    // --- Layer 1 ---
    u32 L1[8]; u32* L1_1 = &L1[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L1[i] = d_add_bytes(d_add_bytes(d_mul2_bytes(O[i]), d_mul2_bytes(T[i])), L0[i]);
    BSWAP4(L1);    aes_encrypt(L1,   (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L1);
    BSWAP4(L1_1);  aes_encrypt(L1_1, (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L1_1);

    // --- Layer 2 ---
    u32 L2[8]; u32* L2_1 = &L2[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) L2[i] = d_add_bytes(d_add_bytes(d_mul2_bytes(O[i]), T[i]), L1[i]);
    BSWAP4(L2);    aes_encrypt(L2,   (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L2);
    BSWAP4(L2_1);  aes_encrypt(L2_1, (u32*)rk_all, t0S, Sbox, warpIdx);  BSWAP4(L2_1);

    u32* res = out + pair_idx * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        res[i] = d_add_bytes(d_add_bytes(d_add_bytes(O[i], L0[i]), L1[i]), d_mul2_bytes(L2[i]));
    }
}

