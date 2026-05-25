// gpu_fields.cuh — Mersenne-61 base field and extension towers for the velox
// CUDA GEMM kernel. Ported from
// async_mpc/fields/cuda/gpu_fields.cuh (M61 sections only — M31 dropped since
// velox's protocol field is Mersenne61 Fp4).
//
// Layout of every struct is chosen to be memory-compatible with the
// corresponding Rust `FieldElement<F>` type so flat byte buffers can be
// reinterpreted on either side without conversion:
//   Fp        ↔ unsigned long long           (8 bytes)
//   Fp2_61    ↔ [FieldElement<M61>; 2]       (16 bytes)
//   Fp4_61    ↔ [FieldElement<Fp2_61>; 2]    (32 bytes — velox's `LargeField`)
//
// Extension tower:  Fp -> Fp2 (i² + 1 = 0) -> Fp4 (w² = 4 + i)

#pragma once
#include <stdint.h>

#define P61 2305843009213693951ULL  // 2^61 - 1

// ============================================================================
// Mersenne-61 base field  (uint64_t)
// ============================================================================

__device__ __forceinline__ unsigned long long fp61_reduce(unsigned long long x) {
    x = (x & P61) + (x >> 61);
    if (x >= P61) x -= P61;
    return x;
}

__device__ __forceinline__ unsigned long long fp61_add(unsigned long long a, unsigned long long b) {
    return fp61_reduce(a + b);
}

__device__ __forceinline__ unsigned long long fp61_sub(unsigned long long a, unsigned long long b) {
    return fp61_reduce(a + P61 - b);
}

// Full 61-bit multiply using hardware 128-bit product.
__device__ __forceinline__ unsigned long long fp61_mul(unsigned long long a, unsigned long long b) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);
    // 2^64 ≡ 8 (mod 2^61-1)
    unsigned long long s = (lo & P61) + (lo >> 61) + (hi << 3);
    s = (s & P61) + (s >> 61);
    if (s >= P61) s -= P61;
    return s;
}

__device__ __forceinline__ unsigned long long fp61_neg(unsigned long long a) {
    return (a == 0) ? 0 : (P61 - a);
}

__device__ __forceinline__ unsigned long long fp61_zero() { return 0ULL; }
__device__ __forceinline__ unsigned long long fp61_one()  { return 1ULL; }

// ============================================================================
// M61 Fp2 = Fp[i] / (i² + 1)       — layout matches Rust [FieldElement<M61>; 2]
// ============================================================================

struct Fp2_61 { unsigned long long re, im; };

__device__ __forceinline__ Fp2_61 fp2_61_add(Fp2_61 a, Fp2_61 b) {
    return { fp61_add(a.re, b.re), fp61_add(a.im, b.im) };
}

__device__ __forceinline__ Fp2_61 fp2_61_sub(Fp2_61 a, Fp2_61 b) {
    return { fp61_sub(a.re, b.re), fp61_sub(a.im, b.im) };
}

__device__ __forceinline__ Fp2_61 fp2_61_neg(Fp2_61 a) {
    return { fp61_neg(a.re), fp61_neg(a.im) };
}

// Karatsuba: (a0+a1*i)(b0+b1*i) = (a0*b0 - a1*b1) + ((a0+a1)(b0+b1) - a0*b0 - a1*b1)*i
__device__ __forceinline__ Fp2_61 fp2_61_mul(Fp2_61 a, Fp2_61 b) {
    unsigned long long a0b0 = fp61_mul(a.re, b.re);
    unsigned long long a1b1 = fp61_mul(a.im, b.im);
    unsigned long long z    = fp61_mul(fp61_add(a.re, a.im), fp61_add(b.re, b.im));
    return { fp61_sub(a0b0, a1b1), fp61_sub(fp61_sub(z, a0b0), a1b1) };
}

__device__ __forceinline__ Fp2_61 fp2_61_zero() { return { 0ULL, 0ULL }; }
__device__ __forceinline__ Fp2_61 fp2_61_one()  { return { 1ULL, 0ULL }; }

// Multiply by the non-residue (4 + i) used in the M61 Fp4 tower.
// (a + b*i)(4 + i) = (4a - b) + (a + 4b)*i
__device__ __forceinline__ Fp2_61 fp2_61_mul_nonresidue(Fp2_61 a) {
    unsigned long long four = 4ULL;
    return {
        fp61_sub(fp61_mul(four, a.re), a.im),
        fp61_add(a.re, fp61_mul(four, a.im))
    };
}

// ============================================================================
// M61 Fp4 = Fp2[w] / (w² − (4+i))  — layout matches Rust [Fp2E; 2]
// ============================================================================

struct Fp4_61 { Fp2_61 c0, c1; };

__device__ __forceinline__ Fp4_61 fp4_61_add(Fp4_61 a, Fp4_61 b) {
    return { fp2_61_add(a.c0, b.c0), fp2_61_add(a.c1, b.c1) };
}

__device__ __forceinline__ Fp4_61 fp4_61_sub(Fp4_61 a, Fp4_61 b) {
    return { fp2_61_sub(a.c0, b.c0), fp2_61_sub(a.c1, b.c1) };
}

__device__ __forceinline__ Fp4_61 fp4_61_neg(Fp4_61 a) {
    return { fp2_61_neg(a.c0), fp2_61_neg(a.c1) };
}

// Karatsuba over Fp2:
// (a0 + a1*w)(b0 + b1*w) = (a0*b0 + (4+i)*a1*b1) + ((a0+a1)(b0+b1) - a0*b0 - a1*b1)*w
__device__ __forceinline__ Fp4_61 fp4_61_mul(Fp4_61 a, Fp4_61 b) {
    Fp2_61 a0b0  = fp2_61_mul(a.c0, b.c0);
    Fp2_61 a1b1  = fp2_61_mul(a.c1, b.c1);
    Fp2_61 cross = fp2_61_sub(
        fp2_61_mul(fp2_61_add(a.c0, a.c1), fp2_61_add(b.c0, b.c1)),
        fp2_61_add(a0b0, a1b1)
    );
    return { fp2_61_add(a0b0, fp2_61_mul_nonresidue(a1b1)), cross };
}

__device__ __forceinline__ Fp4_61 fp4_61_zero() { return { fp2_61_zero(), fp2_61_zero() }; }
__device__ __forceinline__ Fp4_61 fp4_61_one()  { return { fp2_61_one(),  fp2_61_zero() }; }

// ============================================================================
// FieldOps<F> template — uniform interface for the GEMM kernel
// ============================================================================

template<typename F> struct FieldOps;

template<> struct FieldOps<unsigned long long> {
    __device__ static unsigned long long zero()                                          { return fp61_zero(); }
    __device__ static unsigned long long add(unsigned long long a, unsigned long long b) { return fp61_add(a, b); }
    __device__ static unsigned long long mul(unsigned long long a, unsigned long long b) { return fp61_mul(a, b); }
};

template<> struct FieldOps<Fp2_61> {
    __device__ static Fp2_61 zero()                  { return fp2_61_zero(); }
    __device__ static Fp2_61 add(Fp2_61 a, Fp2_61 b) { return fp2_61_add(a, b); }
    __device__ static Fp2_61 mul(Fp2_61 a, Fp2_61 b) { return fp2_61_mul(a, b); }
};

template<> struct FieldOps<Fp4_61> {
    __device__ static Fp4_61 zero()                  { return fp4_61_zero(); }
    __device__ static Fp4_61 add(Fp4_61 a, Fp4_61 b) { return fp4_61_add(a, b); }
    __device__ static Fp4_61 mul(Fp4_61 a, Fp4_61 b) { return fp4_61_mul(a, b); }
};
