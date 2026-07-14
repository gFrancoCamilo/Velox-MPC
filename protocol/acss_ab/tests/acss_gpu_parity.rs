//! CPU ↔ GPU dealer parity tests.
//!
//! Run on a CUDA machine:
//! `cargo test -p acss_ab --features gpu --test acss_gpu_parity -- --test-threads=1`
//!
//! `dealer_core_cpu` and `dealer_core_gpu` take the same deterministic inputs
//! (fixed secret-key map, fixed secrets, fixed nonce/blinding secrets), so
//! every broadcast artifact must match bitwise — except the DZK coefficient
//! vector, which the CPU trims of trailing zeros (Polynomial::new) and is
//! therefore compared semantically.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

use acss_ab::protocol::{dealer_core_cpu, dealer_core_gpu, AcssGemmCache, DealerArtifacts};
use crypto::aes_hash::{HashState, MerkleTree};
use lambdaworks_math::polynomial::Polynomial;
use protocol::{ByteConversion, LargeField};
use types::Replica;

fn hash_state() -> HashState {
    // Same fixed keys as acss_ab::Context::spawn.
    HashState::new([5u8; 16], [29u8; 16], [23u8; 16])
}

fn sec_key_map(n: usize) -> HashMap<Replica, Vec<u8>> {
    (0..n)
        .map(|i| (i as Replica, vec![(i as u8).wrapping_add(1); 32]))
        .collect()
}

fn secrets(count: usize) -> Vec<LargeField> {
    (0..count)
        .map(|i| LargeField::from((i as u64).wrapping_add(12345)))
        .collect()
}

fn run_pair(n: usize, num_polys: usize, cache: &mut AcssGemmCache) -> (DealerArtifacts, DealerArtifacts) {
    let t = (n - 1) / 3;
    let hc = hash_state();
    let map = sec_key_map(n);
    let s = secrets(num_polys);
    let nonce = LargeField::from(11u64);
    let blind = LargeField::from(22u64);
    let blind_nonce = LargeField::from(33u64);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let cpu = rt.block_on(dealer_core_cpu(
        s.clone(),
        nonce.clone(),
        blind.clone(),
        blind_nonce.clone(),
        map.clone(),
        t,
        n,
        &hc,
    ));
    let gpu = dealer_core_gpu(&s, nonce, blind, blind_nonce, map, t, n, &hc, cache)
        .expect("GPU dealer failed");
    (cpu, gpu)
}

fn parse_coeffs(ser: &[Vec<u8>]) -> Vec<LargeField> {
    ser.iter()
        .map(|c| LargeField::from_bytes_be(c).unwrap())
        .collect()
}

fn assert_artifacts_match(cpu: &DealerArtifacts, gpu: &DealerArtifacts, label: &str) {
    assert_eq!(
        cpu.party_wise_shares, gpu.party_wise_shares,
        "{}: party_wise_shares mismatch",
        label
    );
    assert_eq!(
        cpu.nonce_evals_ser, gpu.nonce_evals_ser,
        "{}: nonce_evals mismatch",
        label
    );
    assert_eq!(
        cpu.blinding_nonce_evals_ser, gpu.blinding_nonce_evals_ser,
        "{}: blinding_nonce_evals mismatch",
        label
    );
    assert_eq!(
        cpu.commitments, gpu.commitments,
        "{}: commitments mismatch",
        label
    );
    assert_eq!(
        cpu.blinding_commitments, gpu.blinding_commitments,
        "{}: blinding_commitments mismatch",
        label
    );

    // DZK: semantic comparison (CPU trims trailing-zero coefficients).
    let mut dzk_cpu = parse_coeffs(&cpu.ser_dzk_coeffs);
    let mut dzk_gpu = parse_coeffs(&gpu.ser_dzk_coeffs);
    let len = dzk_cpu.len().max(dzk_gpu.len());
    dzk_cpu.resize(len, LargeField::zero());
    dzk_gpu.resize(len, LargeField::zero());
    assert_eq!(dzk_cpu, dzk_gpu, "{}: dzk coefficients mismatch", label);
}

#[test]
fn dealer_parity_across_sizes() {
    for &(n, num_polys) in &[(4usize, 1usize), (4, 100), (16, 100), (16, 4096)] {
        // Fresh cache per (t, n) shape; reuse across poly counts is tested below.
        let mut cache = AcssGemmCache::new();
        let (cpu, gpu) = run_pair(n, num_polys, &mut cache);
        assert_artifacts_match(&cpu, &gpu, &format!("n={} polys={}", n, num_polys));
    }
}

#[test]
fn dealer_parity_large_batch() {
    // Large batch exercises the canonical-reduction risk (R2 in the plan):
    // any non-canonical limb out of the GEMM chain breaks byte parity here.
    let mut cache = AcssGemmCache::new();
    let (cpu, gpu) = run_pair(16, 100_000, &mut cache);
    assert_artifacts_match(&cpu, &gpu, "n=16 polys=100000");
}

#[test]
fn dealer_parity_with_cache_reuse() {
    // Same cache across three calls with different batch sizes — mirrors the
    // long-lived Context. Buffers must grow/reuse without stale data.
    let mut cache = AcssGemmCache::new();
    for &num_polys in &[64usize, 4096, 128] {
        let (cpu, gpu) = run_pair(16, num_polys, &mut cache);
        assert_artifacts_match(&cpu, &gpu, &format!("cache-reuse polys={}", num_polys));
    }
}

/// The GPU dealer's artifacts must pass the exact acceptance checks
/// `verify_shares` + `evaluate_dzk_poly` run on every receiving party.
#[test]
fn gpu_dealer_passes_cpu_verifier() {
    let n = 16usize;
    let t = (n - 1) / 3;
    let num_polys = 500usize;
    let hc = hash_state();

    let mut cache = AcssGemmCache::new();
    let gpu = dealer_core_gpu(
        &secrets(num_polys),
        LargeField::from(11u64),
        LargeField::from(22u64),
        LargeField::from(33u64),
        sec_key_map(n),
        t,
        n,
        &hc,
        &mut cache,
    )
    .expect("GPU dealer failed");

    // Fiat-Shamir challenge, recomputed exactly as verify_shares does.
    let share_root = MerkleTree::new(gpu.commitments.clone(), &hc).root();
    let blinding_root = MerkleTree::new(gpu.blinding_commitments.clone(), &hc).root();
    let root_comm = hc.hash_two(share_root, blinding_root);
    let root_comm_fe = LargeField::from_bytes_be(&root_comm).unwrap();

    let dzk_poly = Polynomial::new(parse_coeffs(&gpu.ser_dzk_coeffs).as_slice());

    for i in 0..n {
        // Share commitment check (verify_shares).
        let mut appended = Vec::new();
        for share in &gpu.party_wise_shares[i] {
            appended.extend_from_slice(share);
        }
        appended.extend_from_slice(&gpu.nonce_evals_ser[i]);
        assert_eq!(
            hc.do_hash_aes(&appended),
            gpu.commitments[i],
            "party {}: share commitment rejected",
            i
        );

        // DZK check (evaluate_dzk_poly).
        let dzk_point = dzk_poly.evaluate(&LargeField::from((i + 1) as u64));
        let mut agg = LargeField::zero();
        let mut r_pow = root_comm_fe.clone();
        for share in &gpu.party_wise_shares[i] {
            let share_fe = LargeField::from_bytes_be(share).unwrap();
            agg = agg.add(share_fe.mul(r_pow.clone()));
            r_pow = r_pow.mul(root_comm_fe.clone());
        }
        let blinding_share_bytes = dzk_point.sub(agg).to_bytes_be();
        let blinding_hash = hc.hash_two(
            blinding_share_bytes.try_into().unwrap(),
            gpu.blinding_nonce_evals_ser[i].clone().try_into().unwrap(),
        );
        assert_eq!(
            blinding_hash, gpu.blinding_commitments[i],
            "party {}: DZK proof rejected",
            i
        );
    }
}
