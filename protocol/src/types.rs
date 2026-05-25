use lambdaworks_math::field::element::FieldElement;

use crate::mersenne_61::Mersenne61Degree4ExtensionField;

/// Protocol field. 32-byte degree-4 extension of the Mersenne-61 prime field —
/// 4 × u64 little-endian limbs, ~244-bit soundness, byte width matches the prior
/// BN254 element so wire / serialization sizes stay the same.
pub type LargeField = FieldElement<Mersenne61Degree4ExtensionField>;

/// Serialized form of a `LargeField`. Always 32 bytes for Fp4_61.
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

/// Per-share triple emitted by the AVSS layer. Kept exactly as before — only the
/// element width inside `LargeFieldSer` has changed (32 bytes, same as BN254).
pub type AvssShare = (Vec<LargeFieldSer>, LargeFieldSer, LargeFieldSer);
