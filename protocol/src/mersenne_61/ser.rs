// Adapted from async_mpc/fields/src/mersenne_61/ser.rs.
// async_mpc uses its own ByteConversion trait; this version implements lambdaworks's
// stock `ByteConversion` (`to_bytes_be`/`from_bytes_be`) so the M61 types slot into
// velox's existing serialization call sites without further changes.

use lambdaworks_math::{errors::ByteConversionError, field::element::FieldElement};

use crate::byte_conv::ByteConversion;

use super::{
    extensions::{Mersenne61Degree2ExtensionField, Mersenne61Degree4ExtensionField},
    field::Mersenne61Field,
};

impl ByteConversion for FieldElement<Mersenne61Field> {
    fn to_bytes_be(&self) -> Vec<u8> {
        self.representative().to_be_bytes().to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.representative().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; 8] = bytes
            .try_into()
            .map_err(|_| ByteConversionError::FromBEBytesError)?;
        Ok(Self::from(&u64::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; 8] = bytes
            .try_into()
            .map_err(|_| ByteConversionError::FromLEBytesError)?;
        Ok(Self::from(&u64::from_le_bytes(bytes)))
    }
}

impl ByteConversion for FieldElement<Mersenne61Degree2ExtensionField> {
    // Flat 16-byte layout: 2 × 8-byte limbs.
    fn to_bytes_be(&self) -> Vec<u8> {
        let v = self.value();
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&v[0].representative().to_be_bytes());
        out[8..16].copy_from_slice(&v[1].representative().to_be_bytes());
        out.to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        let v = self.value();
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&v[0].representative().to_le_bytes());
        out[8..16].copy_from_slice(&v[1].representative().to_le_bytes());
        out.to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        const BYTES_PER_FIELD: usize = 8;
        const EXPECTED_LEN: usize = BYTES_PER_FIELD * 2;
        if bytes.len() < EXPECTED_LEN {
            return Err(ByteConversionError::FromBEBytesError);
        }
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..EXPECTED_LEN])?;
        Ok(Self::new([x0, x1]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        const BYTES_PER_FIELD: usize = 8;
        const EXPECTED_LEN: usize = BYTES_PER_FIELD * 2;
        if bytes.len() < EXPECTED_LEN {
            return Err(ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..EXPECTED_LEN])?;
        Ok(Self::new([x0, x1]))
    }
}

impl ByteConversion for FieldElement<Mersenne61Degree4ExtensionField> {
    // Flat 32-byte layout: 4 × 8-byte limbs. Same byte width as the old BN254 element.
    fn to_bytes_be(&self) -> Vec<u8> {
        let v = self.value();
        let v0 = v[0].value();
        let v1 = v[1].value();
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&v0[0].representative().to_be_bytes());
        out[8..16].copy_from_slice(&v0[1].representative().to_be_bytes());
        out[16..24].copy_from_slice(&v1[0].representative().to_be_bytes());
        out[24..32].copy_from_slice(&v1[1].representative().to_be_bytes());
        out.to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        let v = self.value();
        let v0 = v[0].value();
        let v1 = v[1].value();
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&v0[0].representative().to_le_bytes());
        out[8..16].copy_from_slice(&v0[1].representative().to_le_bytes());
        out[16..24].copy_from_slice(&v1[0].representative().to_le_bytes());
        out[24..32].copy_from_slice(&v1[1].representative().to_le_bytes());
        out.to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        const BYTES_PER_FIELD: usize = 16;
        const EXPECTED_LEN: usize = BYTES_PER_FIELD * 2;
        if bytes.len() < EXPECTED_LEN {
            return Err(ByteConversionError::FromBEBytesError);
        }
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..EXPECTED_LEN])?;
        Ok(Self::new([x0, x1]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        const BYTES_PER_FIELD: usize = 16;
        const EXPECTED_LEN: usize = BYTES_PER_FIELD * 2;
        if bytes.len() < EXPECTED_LEN {
            return Err(ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..EXPECTED_LEN])?;
        Ok(Self::new([x0, x1]))
    }
}
