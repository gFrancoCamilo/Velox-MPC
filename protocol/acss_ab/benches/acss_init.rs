//! ACSS dealer benchmarks: CPU vs GPU dealer core, plus the commitment-hash
//! primitive (SHA-256 vs AES) that step 1 of the GPU port switched.
//!
//! CPU only:  cargo bench -p acss_ab --bench acss_init
//! With GPU:  cargo bench -p acss_ab --features gpu --bench acss_init
//!
//! Adapted from async_mpc/secret_sharing/acss_ab/benches/acss_init.rs.

use std::collections::HashMap;

use acss_ab::protocol::dealer_core_cpu;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crypto::aes_hash::HashState;
use crypto::hash::do_hash;
use protocol::{ByteConversion, LargeField};
use types::Replica;

#[cfg(feature = "gpu")]
use acss_ab::protocol::{dealer_core_gpu, AcssGemmCache};

const N: usize = 16;
const POLY_COUNTS: &[usize] = &[1024, 16384, 131072];

fn hash_state() -> HashState {
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

fn bench_dealer_cpu(c: &mut Criterion) {
    let t = (N - 1) / 3;
    let hc = hash_state();
    let map = sec_key_map(N);
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("AcssDealerCPU");
    group.sample_size(10);
    for &num_polys in POLY_COUNTS {
        let s = secrets(num_polys);
        group.throughput(Throughput::Elements(num_polys as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_polys), &s, |b, s| {
            b.iter(|| {
                rt.block_on(dealer_core_cpu(
                    s.clone(),
                    LargeField::from(11u64),
                    LargeField::from(22u64),
                    LargeField::from(33u64),
                    map.clone(),
                    t,
                    N,
                    &hc,
                ))
            })
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_dealer_gpu(c: &mut Criterion) {
    let t = (N - 1) / 3;
    let hc = hash_state();
    let map = sec_key_map(N);
    // Shared cache across iterations — matches the long-lived Context, so the
    // steady-state numbers exclude the one-time matrix uploads.
    let mut cache = AcssGemmCache::new();

    let mut group = c.benchmark_group("AcssDealerGPU");
    group.sample_size(10);
    for &num_polys in POLY_COUNTS {
        let s = secrets(num_polys);
        group.throughput(Throughput::Elements(num_polys as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_polys), &s, |b, s| {
            b.iter(|| {
                dealer_core_gpu(
                    s,
                    LargeField::from(11u64),
                    LargeField::from(22u64),
                    LargeField::from(33u64),
                    map.clone(),
                    t,
                    N,
                    &hc,
                    &mut cache,
                )
                .expect("GPU dealer failed")
            })
        });
    }
    group.finish();
}

/// The per-party commitment hash over a realistic row (num_polys shares + one
/// nonce, 32 bytes each): SHA-256 `do_hash` (old wire format) vs the AES-based
/// `do_hash_aes` (new wire format shared with the GPU path).
fn bench_commit_hash(c: &mut Criterion) {
    let hc = hash_state();

    let mut group = c.benchmark_group("CommitHash");
    for &num_polys in POLY_COUNTS {
        let row: Vec<u8> = secrets(num_polys + 1)
            .into_iter()
            .flat_map(|el| el.to_bytes_be())
            .collect();
        group.throughput(Throughput::Bytes(row.len() as u64));
        group.bench_with_input(BenchmarkId::new("sha256", num_polys), &row, |b, row| {
            b.iter(|| do_hash(row))
        });
        group.bench_with_input(BenchmarkId::new("aes", num_polys), &row, |b, row| {
            b.iter(|| hc.do_hash_aes(row))
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_dealer_cpu, bench_dealer_gpu, bench_commit_hash);
#[cfg(not(feature = "gpu"))]
criterion_group!(benches, bench_dealer_cpu, bench_commit_hash);
criterion_main!(benches);
