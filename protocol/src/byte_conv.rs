//! Local re-statement of lambdaworks-math's `ByteConversion` trait.
//!
//! Why we need our own copy: lambdaworks's `FieldElement` isn't `#[fundamental]`,
//! so we can't impl a foreign trait (`lambdaworks_math::traits::ByteConversion`)
//! on a foreign generic (`FieldElement<F>`) even when `F` is a locally-defined
//! field (`Mersenne61Field`, etc.). Defining the trait here lets us implement it
//! for `FieldElement<Mersenne61DegreeNExtensionField>` — orphan rule satisfied
//! because the trait is local. Method surface is byte-for-byte identical to
//! lambdaworks's so existing call sites only need to swap their `use` statement.

use lambdaworks_math::errors::ByteConversionError;

pub trait ByteConversion: Sized {
    fn to_bytes_be(&self) -> Vec<u8>;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>;
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>;
}
