//! Correctness gate for the CUDA GEMM path.
//!
//! Only compiled with `--features gpu`; needs an NVIDIA box + CUDA runtime at
//! test time. For each shape in the matrix below, generates random
//! `LargeField` (Mersenne61 Fp4) inputs, runs both the GPU dispatcher and the
//! pure-CPU reference, and asserts the outputs are byte-equal. Same pattern as
//! `async_mpc/fields/tests/field_gemm.rs`.
//!
//! Run on the GPU server:
//!   cargo test --features gpu -p protocol --test field_gemm_gpu

#![cfg(feature = "gpu")]

use protocol::{
    gpu_ffi::gpu_matrix_matrix_multiply, matrix_matrix_multiply_cpu, rand_field_element,
    LargeField,
};

fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<LargeField>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rand_field_element()).collect())
        .collect()
}

fn assert_gpu_eq_cpu(rows: usize, cols: usize, batch: usize, row_major: bool) {
    let matrix = rand_matrix(rows, cols);
    let vectors = rand_matrix(batch, cols);

    let cpu = matrix_matrix_multiply_cpu(&matrix, &vectors, row_major);
    let gpu = gpu_matrix_matrix_multiply(&matrix, &vectors, row_major);

    assert_eq!(
        cpu.len(),
        gpu.len(),
        "outer length mismatch at {}x{}_b{} row_major={}",
        rows,
        cols,
        batch,
        row_major
    );
    for (i, (cr, gr)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            cr.len(),
            gr.len(),
            "inner length mismatch row {} at {}x{}_b{} row_major={}",
            i,
            rows,
            cols,
            batch,
            row_major
        );
        assert_eq!(
            cr, gr,
            "value mismatch at row {} of {}x{}_b{} row_major={}",
            i, rows, cols, batch, row_major
        );
    }
}

#[test]
fn small_4x8_b16() {
    assert_gpu_eq_cpu(4, 8, 16, true);
    assert_gpu_eq_cpu(4, 8, 16, false);
}

#[test]
fn medium_16x32_b64() {
    assert_gpu_eq_cpu(16, 32, 64, true);
    assert_gpu_eq_cpu(16, 32, 64, false);
}

#[test]
fn async_mpc_reference_64x128_b1024() {
    // Same shape as async_mpc/fields/benches/field_gemm.rs's M61_Fp4 bench so
    // a passing assertion here means the two repos compute the same GEMM.
    assert_gpu_eq_cpu(64, 128, 1024, true);
    assert_gpu_eq_cpu(64, 128, 1024, false);
}
