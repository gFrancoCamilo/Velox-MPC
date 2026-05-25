use std::{collections::HashMap, ops::{Mul, Sub, Add}};

use crypto::hash::do_hash;
use lambdaworks_math::{unsigned_integer::element::UnsignedInteger, polynomial::Polynomial};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::{SeedableRng, RngCore};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use types::Replica;

use crate::LargeField;

pub fn sample_polynomials_from_prf(
    secrets: Vec<LargeField>, 
    sec_key_map: HashMap<Replica, Vec<u8>>, 
    degree: usize,
    is_nonce: bool,
    nonce: u8
)-> Vec<Vec<LargeField>>{
    let tot_evaluations = secrets.len();
    let mut evaluations = Vec::new();
    for secret in secrets{
        evaluations.push(vec![secret]);
    }
    for i in 0..degree{
        let mut sec_key = sec_key_map.get(&(i as Replica)).unwrap().clone();
        if is_nonce{
            sec_key.push(nonce);
        }
        let samples = pseudorandom_lf(&sec_key, tot_evaluations);
        for (i,sample) in samples.into_iter().enumerate() {
            evaluations[i].push(sample);
        }
    }
    evaluations
}

pub async fn generate_evaluation_points(
    evaluations_prf: Vec<Vec<LargeField>>, 
    degree: usize,
    shares_total: usize,
) -> (Vec<Vec<LargeField>>, 
    Vec<Polynomial<LargeField>>
){

    // The first evaluation is always at 0
    let mut evaluation_points = Vec::new();
    evaluation_points.push(LargeField::from(0u64));
    for i in 0..degree{
        evaluation_points.push(LargeField::from((i + 1) as u64));
    }
    
    // Generate coefficients of polynomial and then evaluate the polynomial at n points
    let coefficients: Vec<Polynomial<LargeField>> = evaluations_prf.into_par_iter().map(|evals| {
        return Polynomial::interpolate(evaluation_points.as_slice(), evals.as_slice()).unwrap()
    }).collect();

    // Evaluate the polynomial at n points
    let evaluations_full = coefficients.par_iter().map(|polynomial|{
        let mut eval_vec_ind = Vec::new();
        for index in 0..shares_total{
            eval_vec_ind.push(polynomial.evaluate(&LargeField::from((index + 1) as u64)));
        }
        return eval_vec_ind;
    }).collect();
    (evaluations_full,coefficients)
}

pub async fn generate_evaluation_points_opt(
    evaluations_prf: Vec<Vec<LargeField>>,
    degree: usize,
    shares_total: usize,
) -> (Vec<Vec<LargeField>>,
    Vec<Polynomial<LargeField>>
){

    // The first evaluation is always at 0
    let mut evaluation_points = Vec::new();
    evaluation_points.push(LargeField::from(0u64));
    for i in 0..degree{
        evaluation_points.push(LargeField::from((i + 1) as u64));
    }

    // Generate vandermonde matrix
    let vandermonde = vandermonde_matrix(evaluation_points.clone());
    let inverse_vandermonde_mat = inverse_vandermonde(vandermonde);

    // Two-GEMM Lagrange (mirrors async_mpc/mpc/protocol/verification/compress_tup.rs:191):
    //   Step 1: coeffs_mat = inv_vandermonde · evaluations_prf  →  num_polys × (degree+1).
    //   Step 2: evals_mat  = powers_matrix(share_points, degree+1) · coeffs_mat
    //                       → num_polys × shares_total.
    // `BatchedLagrange` bench: at protocol-realistic n=16, t=5, GEMM is ~1.18× faster
    // than the prior per-poly mat-vec + per-point Horner once num_polys ≥ 1024.
    let coeffs_mat = matrix_matrix_multiply(&inverse_vandermonde_mat, &evaluations_prf, false);

    let share_points: Vec<LargeField> = (1..=shares_total)
        .map(|i| LargeField::from(i as u64))
        .collect();
    let share_powers = powers_matrix(&share_points, degree + 1);
    let evaluations_full = matrix_matrix_multiply(&share_powers, &coeffs_mat, false);

    // The return type wants `Vec<Polynomial<LargeField>>`; reconstruct via Polynomial::new
    // (which trims trailing zeros — semantically identical to the prior path).
    let coefficients: Vec<Polynomial<LargeField>> = coeffs_mat
        .par_iter()
        .map(|row| Polynomial::new(row))
        .collect();

    (evaluations_full, coefficients)
}

pub async fn generate_evaluation_points_fft(
    secrets: Vec<LargeField>,
    degree_poly: usize,
    shares_total: usize,
)-> (Vec<Vec<LargeField>>, 
    Vec<Polynomial<LargeField>>
){
    // For FFT evaluations, first sample coefficients of polynomial and then interpolate all n points
    let coefficients: Vec<Vec<LargeField>> = secrets.into_par_iter().map(|secret| {
        let mut coeffs_single_poly = Vec::new();
        coeffs_single_poly.push(secret);
        for _ in 0..degree_poly{
            coeffs_single_poly.push(rand_field_element());
        }
        return Polynomial::new(&coeffs_single_poly).coefficients;
    }).collect();

    return generate_evaluation_points_opt(coefficients, degree_poly, shares_total).await;
}

// pub async fn generate_evaluation_points_fft(
//     secrets: Vec<LargeField>,
//     degree_poly: usize,
//     shares_total: usize,
// )-> (Vec<Vec<LargeField>>, 
//     Vec<Polynomial<LargeField>>
// ){
//     // For FFT evaluations, first sample coefficients of polynomial and then interpolate all n points
//     let coefficients: Vec<Polynomial<LargeField>> = secrets.into_par_iter().map(|secret| {
//         let mut coeffs_single_poly = Vec::new();
//         coeffs_single_poly.push(secret);
//         for _ in 0..degree_poly{
//             coeffs_single_poly.push(rand_field_element());
//         }
//         return Polynomial::new(&coeffs_single_poly);
//     }).collect();

//     let evaluations = coefficients.par_iter().map(|poly_coeffs|{
//         let poly_evaluations_fft = Polynomial::evaluate_fft::<MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>>(poly_coeffs, 1, Some(shares_total)).unwrap();
//         poly_evaluations_fft
//     }).collect();
//     (evaluations, coefficients)
// }

pub fn pseudorandom_lf(rng_seed: &[u8], num: usize) -> Vec<LargeField> {
    let mut rng = ChaCha20Rng::from_seed(do_hash(rng_seed));
    let mut random_numbers: Vec<LargeField> = Vec::with_capacity(num);
    for _ in 0..num {
        // Fp4_61 element = 4 base-field (Mersenne-61) elements packed as two Fp2 components.
        // Each base element accepts a u64; values ≥ p are folded via `from_u64`.
        let c0 = lambdaworks_math::field::element::FieldElement::<
            crate::mersenne_61::Mersenne61Field,
        >::from(rng.next_u64());
        let c1 = lambdaworks_math::field::element::FieldElement::<
            crate::mersenne_61::Mersenne61Field,
        >::from(rng.next_u64());
        let c2 = lambdaworks_math::field::element::FieldElement::<
            crate::mersenne_61::Mersenne61Field,
        >::from(rng.next_u64());
        let c3 = lambdaworks_math::field::element::FieldElement::<
            crate::mersenne_61::Mersenne61Field,
        >::from(rng.next_u64());
        let lo = crate::mersenne_61::Fp2E::new([c0, c1]);
        let hi = crate::mersenne_61::Fp2E::new([c2, c3]);
        random_numbers.push(LargeField::new([lo, hi]));
    }
    random_numbers
}

pub fn rand_field_element() -> LargeField {
    // Sample 4 independent u64 limbs and fold each into the Mersenne-61 base field.
    let r: [u64; 4] = random();
    let c0 = lambdaworks_math::field::element::FieldElement::<
        crate::mersenne_61::Mersenne61Field,
    >::from(r[0]);
    let c1 = lambdaworks_math::field::element::FieldElement::<
        crate::mersenne_61::Mersenne61Field,
    >::from(r[1]);
    let c2 = lambdaworks_math::field::element::FieldElement::<
        crate::mersenne_61::Mersenne61Field,
    >::from(r[2]);
    let c3 = lambdaworks_math::field::element::FieldElement::<
        crate::mersenne_61::Mersenne61Field,
    >::from(r[3]);
    let lo = crate::mersenne_61::Fp2E::new([c0, c1]);
    let hi = crate::mersenne_61::Fp2E::new([c2, c3]);
    LargeField::new([lo, hi])
}


pub fn interpolate_shares( mut secret_key: Vec<u8>, num_shares: usize, is_nonce: bool, padding: u8) -> Vec<LargeField>{
    if is_nonce{
        secret_key.push(padding);
    }
    let prf_values = pseudorandom_lf(&secret_key, num_shares);
    prf_values
}

pub fn check_if_all_points_lie_on_degree_x_polynomial(eval_points: Vec<LargeField>, polys_vector: Vec<Vec<LargeField>>, degree: usize) -> (bool,Option<Vec<Polynomial<LargeField>>>){
    //log::info!("Checking evaluations on points :{:?}, eval_points: {:?}", eval_points, polys_vector);
    let inverse_vandermonde_mat = inverse_vandermonde(vandermonde_matrix(eval_points[0..degree].to_vec()));

    // Two-GEMM Lagrange: recover coefficients from the first `degree` evaluations, then
    // batch-evaluate every recovered polynomial at the remaining `eval_points[degree..]`
    // for consistency-check against the supplied shares. Same idiom as
    // async_mpc/degree_verification/interpolation.rs:645 + 488/509.
    let prefix_vecs: Vec<Vec<LargeField>> = polys_vector
        .par_iter()
        .map(|points| points[0..degree].to_vec())
        .collect();
    let coeffs_mat = matrix_matrix_multiply(&inverse_vandermonde_mat, &prefix_vecs, false);

    let verify_points: Vec<LargeField> = eval_points[degree..].to_vec();
    let verify_evals = if verify_points.is_empty() {
        // Nothing to check past the first `degree` points — every poly passes by definition.
        vec![Vec::new(); coeffs_mat.len()]
    } else {
        let verify_powers = powers_matrix(&verify_points, degree);
        matrix_matrix_multiply(&verify_powers, &coeffs_mat, false)
    };

    // For each polynomial: does its recovered form match the supplied `points[degree..]`?
    let polys: Vec<Option<Polynomial<LargeField>>> = coeffs_mat
        .par_iter()
        .zip(polys_vector.par_iter())
        .zip(verify_evals.par_iter())
        .map(|((coeffs, points), evals)| {
            let expected = &points[degree..];
            if evals
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| got == want)
            {
                Some(Polynomial::new(coeffs))
            } else {
                None
            }
        })
        .collect();

    let all_polys_positive = polys.par_iter().all(|poly| poly.is_some());
    if all_polys_positive {
        let polys_vec = polys.into_iter().map(|x| x.unwrap()).collect();
        (true, Some(polys_vec))
    } else {
        (false, None)
    }
}


/// Constructs the Vandermonde matrix for a given set of x-values.
pub fn vandermonde_matrix(x_values: Vec<LargeField>) -> Vec<Vec<LargeField>> {
    let n = x_values.len();
    let mut matrix = vec![vec![LargeField::zero(); n]; n];

    for (row, x) in x_values.iter().enumerate() {
        let mut value = LargeField::one();
        for col in 0..n {
            matrix[row][col] = value.clone();
            value = value.mul(x);
        }
    }

    matrix
}

/// Computes the inverse of a Vandermonde matrix modulo prime using Gaussian elimination.
pub fn inverse_vandermonde(matrix: Vec<Vec<LargeField>>) -> Vec<Vec<LargeField>> {
    let n = matrix.len();
    let mut augmented = matrix.clone();

    // Extend the matrix with an identity matrix on the right
    for i in 0..n {
        augmented[i].extend((0..n).map(|j| if i == j { LargeField::one() } else { LargeField::zero() }));
    }

    // Perform Gaussian elimination
    for col in 0..n {
        // Normalize pivot row
        let inv = &augmented[col][col].inv().unwrap();
        for k in col..2 * n {
            augmented[col][k] = augmented[col][k].clone().mul(inv);
        }

        // Eliminate other rows
        for row in 0..n {
            if row != col {
                let factor = augmented[row][col].clone();
                for k in col..2 * n {
                    augmented[row][k] = augmented[row][k].clone().sub(factor.clone().mul(augmented[col][k].clone()));
                }
            }
        }
    }

    // Extract the right half as the inverse
    augmented
        .into_iter()
        .map(|row| row[n..2 * n].to_vec())
        .collect()
}

pub fn matrix_vector_multiply(
    matrix: &Vec<Vec<LargeField>>,
    vector: &Vec<LargeField>,
) -> Vec<LargeField> {
    matrix
        .par_iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(LargeField::zero(), |sum, (a, b)| sum.add(a.mul(b)))
        })
        .collect()
}

/// CPU-only batched matrix-matrix multiply over `LargeField` (Rayon-parallel).
///
/// `matrix` is treated as an (R × C) matrix (R = `matrix.len()`, C = `matrix[0].len()`).
/// `vectors` is a slice of K vectors each of length C — i.e. the right operand viewed as
/// a (K × C) row-major buffer whose rows are the *columns* of the right matrix.
/// The output is the (R × K) product `M · Vᵀ`:
///   - row_major = true  → `output[r][i] = Σ_l matrix[r][l] * vectors[i][l]` (shape R × K).
///   - row_major = false → the same product transposed (shape K × R), useful when callers
///     want `output[i][r]` (per-vector outputs grouped first), matching async_mpc's layout.
///
/// Layout is identical to `async_mpc/fields/src/poly.rs::matrix_matrix_multiply_cpu` so the
/// two projects' benchmarks are directly comparable.
///
/// Dispatcher: with `--features gpu`, routes large enough inputs to the CUDA kernel
/// (see `gpu_ffi::gpu_matrix_matrix_multiply`); otherwise (or for small inputs that
/// would be dominated by PCIe upload overhead) calls `matrix_matrix_multiply_cpu`
/// directly. The CPU path stays the canonical reference and is always callable.
pub fn matrix_matrix_multiply(
    matrix: &[Vec<LargeField>],
    vectors: &[Vec<LargeField>],
    row_major: bool,
) -> Vec<Vec<LargeField>> {
    #[cfg(feature = "gpu")]
    {
        // Mirrors async_mpc's small-input bailout. Below this work threshold the
        // device upload cost dominates the actual arithmetic. Tune once the kernel
        // exists.
        const GPU_THRESHOLD: usize = 50_000;
        let rows = matrix.len();
        let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
        let work = rows.saturating_mul(cols).saturating_mul(vectors.len());
        if work >= GPU_THRESHOLD {
            return crate::gpu_ffi::gpu_matrix_matrix_multiply(matrix, vectors, row_major);
        }
    }
    matrix_matrix_multiply_cpu(matrix, vectors, row_major)
}

/// CPU implementation — Rayon-parallel over rows. Always available regardless of
/// the `gpu` feature; callers needing to force CPU (e.g. the bench's reference
/// path) should call this directly.
pub fn matrix_matrix_multiply_cpu(
    matrix: &[Vec<LargeField>],
    vectors: &[Vec<LargeField>],
    row_major: bool,
) -> Vec<Vec<LargeField>> {
    let k = vectors.len();
    if k == 0 || matrix.is_empty() {
        return Vec::new();
    }

    let cols = matrix[0].len();
    if vectors.iter().any(|v| v.len() != cols) {
        log::error!(
            "matrix_matrix_multiply_cpu: matrix column count ({}) does not match vector lengths {:?}",
            cols,
            vectors.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        return Vec::new();
    }

    let results: Vec<Vec<LargeField>> = matrix
        .par_iter()
        .map(|m_row| {
            let mut row_results = vec![LargeField::zero(); k];
            for i in 0..k {
                let m_col = &vectors[i];
                let mut sum = LargeField::zero();
                for l in 0..m_row.len() {
                    // Reference-reference multiply — matches async_mpc verbatim and avoids
                    // the 32-byte per-iteration clone of m_row[l] the previous form did.
                    sum += &m_row[l] * &m_col[l];
                }
                row_results[i] = sum;
            }
            row_results
        })
        .collect();

    if row_major {
        results
    } else {
        transpose(results)
    }
}

/// Build a (points.len() × degree) "powers" matrix where row `i` is
/// `[1, points[i], points[i]^2, …, points[i]^{degree-1}]`.
///
/// Used wherever many polynomials are evaluated at many points: given coefficient matrix
/// `C` (degree × num_polys, column-major over polys), `powers_matrix · C` yields all
/// evaluations in a single GEMM. This is the Vandermonde restricted to the first `degree`
/// columns, mirroring the async_mpc pattern at `fields/src/poly.rs::lagrange_interpolate_par`.
pub fn powers_matrix(points: &[LargeField], degree: usize) -> Vec<Vec<LargeField>> {
    points
        .par_iter()
        .map(|p| {
            let mut row = Vec::with_capacity(degree);
            let mut power = LargeField::one();
            for _ in 0..degree {
                row.push(power.clone());
                power = power.mul(p);
            }
            row
        })
        .collect()
}

/// Transpose a rectangular matrix stored as `Vec<Vec<LargeField>>`.
pub fn transpose(matrix: Vec<Vec<LargeField>>) -> Vec<Vec<LargeField>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    (0..cols)
        .into_par_iter()
        .map(|j| (0..rows).map(|i| matrix[i][j].clone()).collect())
        .collect()
}