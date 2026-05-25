//! Benchmarks for Velox-MPC's GEMM over Mersenne61 Fp4, plus protocol-shaped comparisons.
//!
//! Each `bench_*_compare` group also performs a correctness check: the scalar
//! reference (the existing per-element loop) and the new `matrix_matrix_multiply`
//! path must produce byte-identical outputs before either is timed. This is the
//! gate before any call-site rewrite.
//!
//! Sizes mirror `async_mpc/fields/benches/field_gemm.rs` (rows=64, cols=128, batch=1024)
//! so the two repos' numbers are directly comparable.
//!
//! Run with: `cargo bench --bench field_gemm -p protocol`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::polynomial::Polynomial;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use protocol::{
    inverse_vandermonde, matrix_matrix_multiply, matrix_vector_multiply, powers_matrix,
    rand_field_element, vandermonde_matrix, LargeField,
};

// ---------------------------------------------------------------------------
// Data builders
// ---------------------------------------------------------------------------

fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<LargeField>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rand_field_element()).collect())
        .collect()
}

fn rand_points(n: usize) -> Vec<LargeField> {
    (1..=n).map(|i| LargeField::from(i as u64)).collect()
}

// ---------------------------------------------------------------------------
// Bench 1: raw GEMM at async_mpc's reference shape (apples-to-apples)
// ---------------------------------------------------------------------------

fn bench_raw_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("FieldGEMM/CPU/M61_Fp4");
    let (rows, cols, batch) = (64usize, 128usize, 1024usize);

    let matrix = rand_matrix(rows, cols);
    let vectors = rand_matrix(batch, cols);

    group.bench_with_input(
        BenchmarkId::new("matrix_matrix_multiply", format!("{rows}x{cols}_b{batch}")),
        &(&matrix, &vectors),
        |b, (m, v)| b.iter(|| matrix_matrix_multiply(black_box(m), black_box(v), false)),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 2: batched polynomial evaluation
//   m polynomials of degree d evaluated at n points
//   Scalar reference: Polynomial::new(coeffs).evaluate(point) per (poly, point)
//   GEMM:            powers_matrix(points, d) · coeffs_mat
// ---------------------------------------------------------------------------

fn scalar_batched_poly_eval(
    coeffs: &[Vec<LargeField>],
    points: &[LargeField],
) -> Vec<Vec<LargeField>> {
    // output[r][i] = poly_i(points[r])
    points
        .iter()
        .map(|p| {
            coeffs
                .iter()
                .map(|c| Polynomial::new(c).evaluate(p))
                .collect()
        })
        .collect()
}

fn bench_batched_poly_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("FieldGEMM/Protocol/BatchedPolyEval");

    for &(m, d, n) in &[(64usize, 32usize, 16usize), (256, 32, 64), (256, 32, 256)] {
        let coeffs = rand_matrix(m, d);
        let points = rand_points(n);

        // Correctness gate: scalar reference must match GEMM output.
        let scalar_out = scalar_batched_poly_eval(&coeffs, &points);
        let pm = powers_matrix(&points, d);
        let gemm_out = matrix_matrix_multiply(&pm, &coeffs, true);
        assert_eq!(
            scalar_out, gemm_out,
            "batched poly eval mismatch (m={m}, d={d}, n={n})"
        );

        let id = format!("m{m}_d{d}_n{n}");

        group.bench_with_input(
            BenchmarkId::new("scalar", &id),
            &(&coeffs, &points),
            |b, (c_, p_)| b.iter(|| scalar_batched_poly_eval(black_box(c_), black_box(p_))),
        );

        group.bench_with_input(
            BenchmarkId::new("gemm", &id),
            &(&pm, &coeffs),
            |b, (pm_, c_)| b.iter(|| matrix_matrix_multiply(black_box(pm_), black_box(c_), true)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 3: batched Shamir reconstruction
//   k polynomials of degree t reconstructed from (t+1) shares each.
//   Scalar reference: per-poly matrix_vector_multiply(inv_vdm, shares)
//   GEMM:            inv_vdm · S  (single call for all k polynomials)
// ---------------------------------------------------------------------------

fn scalar_batched_recon(
    inv_vdm: &Vec<Vec<LargeField>>,
    shares_per_poly: &[Vec<LargeField>],
) -> Vec<Vec<LargeField>> {
    shares_per_poly
        .into_par_iter()
        .map(|s| matrix_vector_multiply(inv_vdm, s))
        .collect()
}

fn bench_batched_recon(c: &mut Criterion) {
    let mut group = c.benchmark_group("FieldGEMM/Protocol/BatchedRecon");

    for &(k, t) in &[(64usize, 5usize), (256, 5), (256, 10)] {
        let degree = t + 1;
        let points = rand_points(degree);
        let vdm = vandermonde_matrix(points.clone());
        let inv_vdm = inverse_vandermonde(vdm);

        // shares_per_poly[i] = evaluations of poly_i at `points` (length = degree)
        let shares_per_poly = rand_matrix(k, degree);

        // Scalar: per-poly mat-vec → Vec<Vec<coeff>>; for GEMM we want the same shape.
        // GEMM with row_major=false yields (k × degree) coefficient matrix.
        let scalar_out = scalar_batched_recon(&inv_vdm, &shares_per_poly);
        let gemm_out = matrix_matrix_multiply(&inv_vdm, &shares_per_poly, false);
        assert_eq!(
            scalar_out, gemm_out,
            "batched recon mismatch (k={k}, t={t})"
        );

        let id = format!("k{k}_t{t}");

        group.bench_with_input(
            BenchmarkId::new("scalar", &id),
            &(&inv_vdm, &shares_per_poly),
            |b, (iv, s)| b.iter(|| scalar_batched_recon(black_box(iv), black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("gemm", &id),
            &(&inv_vdm, &shares_per_poly),
            |b, (iv, s)| b.iter(|| matrix_matrix_multiply(black_box(iv), black_box(s), false)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 4: per-party polynomial evaluation across many chunks (lin_mult shape)
//   For each of `chunks` independent z_vectors of length `2t+1`, evaluate at
//   the `n` party points. This is the shape that just got lifted in
//   `protocol/mpc/src/protocol/multiplication/lin_mult.rs`.
//   Scalar reference: per-chunk Polynomial::new(z).evaluate(point_p) loop
//   GEMM:            powers_matrix(party_points, 2t+1) · z_vector per chunk
// ---------------------------------------------------------------------------

fn scalar_party_eval(
    z_vectors: &[Vec<LargeField>],
    party_points: &[LargeField],
) -> Vec<Vec<LargeField>> {
    // output[p][chunk] = poly(z_vectors[chunk]).evaluate(party_points[p])
    party_points
        .iter()
        .map(|pt| {
            z_vectors
                .iter()
                .map(|z| Polynomial::new(z).evaluate(pt))
                .collect()
        })
        .collect()
}

fn gemm_party_eval(
    z_vectors: &[Vec<LargeField>],
    party_powers: &Vec<Vec<LargeField>>,
) -> Vec<Vec<LargeField>> {
    // One GEMM for all chunks: party_powers (n × (2t+1)) · z_vectors → (n × chunks).
    // This is the shape lin_mult.rs uses — collect every chunk's z_vector first, then
    // one GEMM evaluates them all at the n party points. Matches the BatchedPolyEval
    // pattern (which already won ~5×) instead of issuing one GEMM per chunk (which
    // pays Rayon setup overhead each call and lost ~6×).
    matrix_matrix_multiply(party_powers, z_vectors, true)
}

fn bench_batched_party_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("FieldGEMM/Protocol/BatchedPartyEval");

    // Match real protocol shapes: n=16 parties, 2t+1=11 (so t=5).
    let n = 16usize;
    let z_len = 11usize;
    let party_points = rand_points(n);
    let party_powers = powers_matrix(&party_points, z_len);

    for &chunks in &[64usize, 256, 1024] {
        let z_vectors = rand_matrix(chunks, z_len);

        // Correctness gate.
        let scalar_out = scalar_party_eval(&z_vectors, &party_points);
        let gemm_out = gemm_party_eval(&z_vectors, &party_powers);
        assert_eq!(
            scalar_out, gemm_out,
            "batched party eval mismatch (chunks={chunks})"
        );

        let id = format!("n{n}_d{z_len}_chunks{chunks}");

        group.bench_with_input(
            BenchmarkId::new("scalar", &id),
            &(&z_vectors, &party_points),
            |b, (z, pp)| b.iter(|| scalar_party_eval(black_box(z), black_box(pp))),
        );

        group.bench_with_input(
            BenchmarkId::new("gemm", &id),
            &(&z_vectors, &party_powers),
            |b, (z, pm)| b.iter(|| gemm_party_eval(black_box(z), black_box(pm))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 5: full Lagrange interpolation pipeline (generate_evaluation_points_opt shape)
//   Given num_polys vectors of (degree+1) evaluations of distinct polynomials,
//   recover their coefficients then evaluate each at `shares_total` party points.
//   Scalar reference: matches protocol/src/poly.rs:88-100 line-for-line —
//     per-poly matrix_vector_multiply(&inv_vdm, &evals) then per-point Polynomial::evaluate
//   GEMM:           two GEMMs — inv_vdm · evals_mat → coeffs_mat, then powers · coeffs_mat → evals
// ---------------------------------------------------------------------------

fn scalar_lagrange_eval(
    evaluations_prf: &[Vec<LargeField>],
    inv_vdm: &Vec<Vec<LargeField>>,
    share_points: &[LargeField],
) -> Vec<Vec<LargeField>> {
    // Step 1: per-poly coefficient recovery (rayon over polys)
    let coefficients: Vec<Polynomial<LargeField>> = evaluations_prf
        .par_iter()
        .map(|evals| {
            let c = matrix_vector_multiply(inv_vdm, evals);
            Polynomial::new(&c)
        })
        .collect();
    // Step 2: per-poly evaluation at share_points (rayon over polys, rayon inside)
    coefficients
        .par_iter()
        .map(|poly| {
            share_points
                .par_iter()
                .map(|p| poly.evaluate(p))
                .collect()
        })
        .collect()
}

fn gemm_lagrange_eval(
    evaluations_prf: &[Vec<LargeField>],
    inv_vdm: &Vec<Vec<LargeField>>,
    share_powers: &Vec<Vec<LargeField>>,
) -> Vec<Vec<LargeField>> {
    // Step 1: coeffs_mat[i][r] = Σ_l inv_vdm[r][l] * evaluations_prf[i][l]
    //   inv_vdm is (degree+1)×(degree+1); evaluations_prf is num_polys vectors of length (degree+1).
    //   row_major=false → output shape num_polys × (degree+1) (each row is one poly's coefficients).
    let coeffs_mat = matrix_matrix_multiply(inv_vdm, evaluations_prf, false);
    // Step 2: evals_mat[i][p] = Σ_l share_powers[p][l] * coeffs_mat[i][l]
    //   share_powers is shares_total × (degree+1); coeffs_mat is num_polys vectors of length (degree+1).
    //   row_major=false → output shape num_polys × shares_total.
    matrix_matrix_multiply(share_powers, &coeffs_mat, false)
}

fn bench_batched_lagrange(c: &mut Criterion) {
    let mut group = c.benchmark_group("FieldGEMM/Protocol/BatchedLagrange");

    // Fixed protocol shape: t=5, n=16. Sweep num_polys to find the GEMM breakeven.
    // degree+1 = t+1 = 6 (interpolation degree of the Shamir polynomial).
    let degree_plus_one = 6usize;
    let n = 16usize;

    // Same eval_points layout as generate_evaluation_points_opt: [0, 1, …, degree].
    let mut eval_points = Vec::with_capacity(degree_plus_one);
    eval_points.push(LargeField::from(0u64));
    for i in 1..degree_plus_one {
        eval_points.push(LargeField::from(i as u64));
    }
    let inv_vdm = inverse_vandermonde(vandermonde_matrix(eval_points.clone()));

    // Share points = 1, 2, …, n (mirroring `(0..shares_total).map(|i| from(i+1))` in poly.rs:62/99).
    let share_points: Vec<LargeField> = (1..=n).map(|i| LargeField::from(i as u64)).collect();
    let share_powers = powers_matrix(&share_points, degree_plus_one);

    for &num_polys in &[16usize, 64, 256, 1024, 4096] {
        let evaluations_prf = rand_matrix(num_polys, degree_plus_one);

        // Correctness gate: scalar reference must match GEMM byte-for-byte.
        let scalar_out = scalar_lagrange_eval(&evaluations_prf, &inv_vdm, &share_points);
        let gemm_out = gemm_lagrange_eval(&evaluations_prf, &inv_vdm, &share_powers);
        assert_eq!(
            scalar_out, gemm_out,
            "batched lagrange mismatch (num_polys={num_polys})"
        );

        let id = format!("polys{num_polys}_d{}_n{n}", degree_plus_one - 1);

        group.bench_with_input(
            BenchmarkId::new("scalar", &id),
            &(&evaluations_prf, &inv_vdm, &share_points),
            |b, (e, iv, sp)| b.iter(|| scalar_lagrange_eval(black_box(e), black_box(iv), black_box(sp))),
        );

        group.bench_with_input(
            BenchmarkId::new("gemm", &id),
            &(&evaluations_prf, &inv_vdm, &share_powers),
            |b, (e, iv, pw)| b.iter(|| gemm_lagrange_eval(black_box(e), black_box(iv), black_box(pw))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 6 (GPU only): raw GEMM at async_mpc's reference shape, M61_Fp4 on the
// CUDA path. Same `(rows, cols, batch) = (64, 128, 1024)` as the CPU bench, so
// numbers line up 1:1 against async_mpc's M61_Fp4 bench group.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn bench_raw_gemm_gpu(c: &mut Criterion) {
    use protocol::gpu_ffi::gpu_matrix_matrix_multiply;

    let mut group = c.benchmark_group("FieldGEMM/GPU/M61_Fp4");
    let (rows, cols, batch) = (64usize, 128usize, 1024usize);

    let matrix = rand_matrix(rows, cols);
    let vectors = rand_matrix(batch, cols);

    // Correctness gate: GPU output must byte-equal the CPU reference.
    let cpu = matrix_matrix_multiply(&matrix, &vectors, false);
    let gpu = gpu_matrix_matrix_multiply(&matrix, &vectors, false);
    assert_eq!(cpu, gpu, "gpu_matrix_matrix_multiply mismatch at bench shape");

    group.bench_with_input(
        BenchmarkId::new("gpu_matrix_matrix_multiply", format!("{rows}x{cols}_b{batch}")),
        &(&matrix, &vectors),
        |b, (m, v)| b.iter(|| gpu_matrix_matrix_multiply(black_box(m), black_box(v), false)),
    );

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_raw_gemm_gpu(_c: &mut Criterion) {
    // No-op when the GPU feature is off so the criterion_group! list below
    // stays static and the rest of the benches still run on CPU-only hosts.
}

criterion_group!(
    benches,
    bench_raw_gemm,
    bench_batched_poly_eval,
    bench_batched_recon,
    bench_batched_party_eval,
    bench_batched_lagrange,
    bench_raw_gemm_gpu
);
criterion_main!(benches);
