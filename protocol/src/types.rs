use lambdaworks_math::field::element::FieldElement;

use crate::mersenne_61::Mersenne61Field;

/// Protocol field. The Mersenne-61 prime field (p = 2^61 - 1) — a single u64 limb,
/// 8 bytes on the wire.
///
/// Previously this was the degree-4 extension `Mersenne61Degree4ExtensionField`
/// (32 bytes, ~244-bit soundness). The base field is 4x smaller in memory and an
/// Fp multiply is a single `u128` reduction rather than the 9 base multiplies the
/// Fp4 Karatsuba tower costs, which is what makes the NN-inference workload's
/// ~10^10 multiplications tractable.
///
/// Trade-off: statistical soundness of any single field-element check drops to
/// ~2^-61. That is irrelevant while the tuple-verification phase is disabled
/// (see `Context::verification_enabled`), but a restored verification phase must
/// either repeat its checks or run them over an extension of this field.
pub type LargeField = FieldElement<Mersenne61Field>;

/// Serialized form of a `LargeField`. Always 8 bytes for Mersenne-61.
pub type LargeFieldSer = Vec<u8>;

/// Roots-of-unity stub used by share-point selection. Mersenne61 has no FFT
/// support in lambdaworks (no MontgomeryBackend) — this just hands back the
/// non-FFT party-id-as-field-element points, mirroring the previous behaviour
/// in the `!use_fft` branch.
pub fn gen_roots_of_unity(n: usize) -> Vec<LargeField> {
    (1..n + 1)
        .into_iter()
        .map(|x| LargeField::from(x as u64))
        .collect()
}

/// Per-share triple emitted by the AVSS layer.
pub type AvssShare = (Vec<LargeFieldSer>, LargeFieldSer, LargeFieldSer);

/// Widen a serialized field element to the 32-byte width the Merkle / hash layer
/// (`crypto::hash::Hash`) expects.
///
/// `LargeField` serializes to 8 bytes for Mersenne-61, so the value is
/// right-aligned in a zero-padded 32-byte buffer. Injective over field elements,
/// which is all a commitment needs. Before the field switch these were the same
/// width and call sites did a bare `try_into()`, which now fails at runtime.
pub fn field_bytes_to_hash_input(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let n = bytes.len().min(32);
    out[32 - n..].copy_from_slice(&bytes[bytes.len() - n..]);
    out
}

/// Derive a field element from a hash digest (Fiat-Shamir challenge).
///
/// Horner over 7-byte big-endian limbs: every byte of the digest contributes,
/// and each limb is `< 2^56 < p`, so `LargeField::from(u64)` never sees a value
/// that could overflow its `x + 1` reduction step. Truncating to the first 8
/// bytes would work too but would throw away three quarters of the digest.
pub fn hash_to_field(hash: &[u8]) -> LargeField {
    let radix = LargeField::from(1u64 << 56);
    let mut acc = LargeField::zero();
    for chunk in hash.chunks(7) {
        let mut buf = [0u8; 8];
        buf[8 - chunk.len()..].copy_from_slice(chunk);
        acc = acc * &radix + LargeField::from(u64::from_be_bytes(buf));
    }
    acc
}
