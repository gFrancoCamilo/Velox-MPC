//! Unit tests for the ACSS CUDA kernels and the D2D GEMM chain.
//!
//! Run on a CUDA machine: `cargo test -p protocol --features gpu`.
//! Each test pins one LE↔BE conversion point of the GPU dealer pipeline.

#![cfg(feature = "gpu")]

use protocol::gpu_acss_ffi::{append_nonce, compute_dzk_coeffs, pack_party_shares};
use protocol::gpu_mem::{device_to_host, DeviceBuffer};
use protocol::{
    matrix_matrix_multiply_cpu, rand_field_element, read_elem_bytes, ByteConversion, LargeField,
    PreparedFieldGemm,
};

const FIELD_BYTES: usize = 32;
const LIMB_BYTES: usize = 8;

/// Flatten rows into the raw native-LE CUDA layout (32 bytes per element).
fn flatten_rows_le(rows: &[Vec<LargeField>]) -> Vec<u8> {
    let mut flat = Vec::new();
    for row in rows {
        for el in row {
            let ptr = el as *const LargeField as *const u8;
            flat.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, FIELD_BYTES) });
        }
    }
    flat
}

fn random_matrix(rows: usize, cols: usize) -> Vec<Vec<LargeField>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rand_field_element()).collect())
        .collect()
}

/// pack_party_shares(t=0) + append_nonce must reproduce the CPU dealer's
/// per-party row: concat_j evals[j][i].to_bytes_be() || nonce[i].to_bytes_be().
#[test]
fn pack_and_append_match_to_bytes_be() {
    let num_polys = 257; // deliberately not a multiple of the block size
    let n = 16;
    let party_stride = num_polys * FIELD_BYTES + FIELD_BYTES;

    let evals = random_matrix(num_polys, n); // [poly][party]
    let nonce: Vec<LargeField> = (0..n).map(|_| rand_field_element()).collect();

    let mut d_evals = DeviceBuffer::new();
    d_evals.copy_from_host(&flatten_rows_le(&evals)).unwrap();
    let mut d_nonce = DeviceBuffer::new();
    d_nonce.copy_from_host(&flatten_rows_le(&[nonce.clone()])).unwrap();
    let mut d_out = DeviceBuffer::new();
    d_out.ensure(n * party_stride).unwrap();

    unsafe {
        pack_party_shares(
            std::ptr::null(),
            d_evals.as_ptr(),
            d_out.as_mut_ptr(),
            0,
            n,
            num_polys,
            party_stride,
            FIELD_BYTES,
            LIMB_BYTES,
        )
        .unwrap();
        append_nonce(
            d_nonce.as_ptr(),
            d_out.as_mut_ptr(),
            n,
            party_stride,
            num_polys * FIELD_BYTES,
        )
        .unwrap();
    }

    let mut rows = vec![0u8; n * party_stride];
    d_out.copy_to_host(&mut rows, n * party_stride).unwrap();

    for i in 0..n {
        let mut expected = Vec::with_capacity(party_stride);
        for j in 0..num_polys {
            expected.extend(evals[j][i].clone().to_bytes_be());
        }
        expected.extend(nonce[i].clone().to_bytes_be());
        assert_eq!(
            &rows[i * party_stride..(i + 1) * party_stride],
            expected.as_slice(),
            "party {} row mismatch",
            i
        );
    }
}

/// compute_dzk_coeffs must reproduce the CPU combination
/// dzk[c] = blind[c] + Σ_j coeffs[j][c] · powers[j], serialized big-endian.
#[test]
fn dzk_coeffs_kernel_matches_cpu() {
    let num_polys = 1000;
    let n_coeffs = 6; // t + 1

    let coeffs = random_matrix(num_polys, n_coeffs); // [poly][coeff]
    let blind: Vec<LargeField> = (0..n_coeffs).map(|_| rand_field_element()).collect();
    let r = rand_field_element();
    let mut powers = Vec::with_capacity(num_polys);
    let mut acc = r.clone();
    for _ in 0..num_polys {
        powers.push(acc.clone());
        acc = acc * r.clone();
    }

    let mut d_coeffs = DeviceBuffer::new();
    d_coeffs.copy_from_host(&flatten_rows_le(&coeffs)).unwrap();
    let mut d_blind = DeviceBuffer::new();
    d_blind.copy_from_host(&flatten_rows_le(&[blind.clone()])).unwrap();
    let mut d_powers = DeviceBuffer::new();
    d_powers.copy_from_host(&flatten_rows_le(&[powers.clone()])).unwrap();
    let mut d_out = DeviceBuffer::new();
    d_out.ensure(n_coeffs * FIELD_BYTES).unwrap();

    unsafe {
        compute_dzk_coeffs(
            d_coeffs.as_ptr(),
            d_blind.as_ptr(),
            d_powers.as_ptr(),
            d_out.as_mut_ptr(),
            num_polys,
            n_coeffs,
        )
        .unwrap();
    }
    let mut out = vec![0u8; n_coeffs * FIELD_BYTES];
    d_out.copy_to_host(&mut out, n_coeffs * FIELD_BYTES).unwrap();

    for c in 0..n_coeffs {
        let mut expected = blind[c].clone();
        for j in 0..num_polys {
            expected = expected + coeffs[j][c].clone() * powers[j].clone();
        }
        assert_eq!(
            &out[c * FIELD_BYTES..(c + 1) * FIELD_BYTES],
            expected.to_bytes_be().as_slice(),
            "dzk coefficient {} mismatch",
            c
        );
    }
}

/// A two-GEMM D2D chain (output of one context fed as device input of the
/// next) must match the CPU matrix_matrix_multiply chain bit-for-bit.
#[test]
fn d2d_gemm_chain_matches_cpu() {
    let t = 5;
    let n = 16;
    let batch = 512;

    let m1 = random_matrix(t + 1, t + 1); // interpolation-shaped
    let m2 = random_matrix(n, t + 1); // evaluation-shaped
    let input = random_matrix(batch, t + 1);

    let g1 = PreparedFieldGemm::new(&m1).unwrap();
    let g2 = PreparedFieldGemm::new(&m2).unwrap();

    let mut d_in = DeviceBuffer::new();
    d_in.copy_from_host(&flatten_rows_le(&input)).unwrap();

    let (out1_gpu, out2_gpu) = unsafe {
        let d1 = g1.multiply_d2d(d_in.as_ptr(), batch).unwrap();
        // Read the intermediate BEFORE the second GEMM overwrites nothing —
        // d1 lives in g1's buffer, g2 writes to its own, so both stay valid.
        let d2 = g2.multiply_d2d(d1, batch).unwrap();
        let len1 = batch * (t + 1) * FIELD_BYTES;
        let mut buf1 = vec![0u8; len1];
        device_to_host(d1, &mut buf1, len1).unwrap();
        let len2 = batch * n * FIELD_BYTES;
        let mut buf2 = vec![0u8; len2];
        device_to_host(d2, &mut buf2, len2).unwrap();
        (buf1, buf2)
    };

    let out1_cpu = matrix_matrix_multiply_cpu(&m1, &input, false);
    let out2_cpu = matrix_matrix_multiply_cpu(&m2, &out1_cpu, false);

    for i in 0..batch {
        for c in 0..t + 1 {
            let got = read_elem_bytes(
                &out1_gpu[(i * (t + 1) + c) * FIELD_BYTES..(i * (t + 1) + c + 1) * FIELD_BYTES],
            );
            assert_eq!(got, out1_cpu[i][c], "GEMM1 mismatch at ({}, {})", i, c);
        }
        for p in 0..n {
            let got = read_elem_bytes(
                &out2_gpu[(i * n + p) * FIELD_BYTES..(i * n + p + 1) * FIELD_BYTES],
            );
            assert_eq!(got, out2_cpu[i][p], "GEMM2 mismatch at ({}, {})", i, p);
        }
    }
}
