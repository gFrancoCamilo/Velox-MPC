// Ported verbatim (sans tests) from async_mpc/fields/src/mersenne_61/field.rs.
// Mersenne-61 prime field implementation following the lambdaworks IsField trait.

use std::ops::Deref;

use lambdaworks_math::{
    errors::CreationError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsField, IsPrimeField},
    },
};

/// Represents a 61 bit integer value
/// Invariants:
///      62nd-64th bits are clear
///      n < MODULUS
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Mersenne61Field;

pub struct FE1(FieldElement<Mersenne61Field>);

impl Mersenne61Field {
    #[inline(always)]
    fn weak_reduce(n: u64) -> u64 {
        // To reduce 'n' to 61 bits we clear its MSBs, then add them back in reduced form.
        let msb = n >> 61;
        let res = (n & MERSENNE_61_PRIME_FIELD_ORDER) + msb;

        // assert result fits within 62 bits at most (may need one more reduction)
        debug_assert!((res >> 62) == 0);
        res
    }

    #[inline(always)]
    fn as_representative(n: &u64) -> u64 {
        if *n == MERSENNE_61_PRIME_FIELD_ORDER {
            0
        } else {
            *n
        }
    }

    #[inline]
    pub fn sum<I: Iterator<Item = <Self as IsField>::BaseType>>(
        iter: I,
    ) -> <Self as IsField>::BaseType {
        // Delayed reduction
        Self::from_u128(iter.map(|x| x as u128).sum::<u128>())
    }

    /// Computes a * 2^k, with 0 < k < 61
    #[inline(always)]
    pub fn mul_power_two(a: u64, k: u64) -> u64 {
        let msb = (a & (u64::MAX << (61 - k))) >> (61 - k); // The k + 1 msb shifted right.
        let lsb = (a & (u64::MAX >> (k + 3))) << k; // The 61 - k lsb shifted left.
        Self::weak_reduce(msb + lsb)
    }

    #[inline]
    pub fn pow_2(a: &u64, order: u64) -> u64 {
        let mut res = *a;
        (0..order).for_each(|_| res = Self::square(&res));
        res
    }

    /// Computes 2a^2 - 1
    #[inline(always)]
    pub fn two_square_minus_one(a: &u64) -> u64 {
        if *a == 0 {
            MERSENNE_61_PRIME_FIELD_ORDER - 1
        } else {
            Self::from_u128(((u128::from(*a) * u128::from(*a)) << 1) - 1)
        }
    }

    /// Reduces a u128 value modulo the Mersenne-61 prime.
    #[inline(always)]
    pub fn from_u128(x: u128) -> u64 {
        // Split into chunks of 61 bits and sum them
        ((((((x + 1) >> 61) + x + 1) >> 61) + x) & (MERSENNE_61_PRIME_FIELD_ORDER as u128)) as u64
    }
}

pub const MERSENNE_61_PRIME_FIELD_ORDER: u64 = (1u64 << 61) - 1;

//NOTE: This implementation was inspired by and borrows from the work done by the Lambdaworks team
// https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/mersenne31/field.rs
impl IsField for Mersenne61Field {
    type BaseType = u64;

    /// Returns the sum of `a` and `b`.
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        // a + b has at most 62 bits, so we can use weak_reduce.
        Self::weak_reduce(a + b)
    }

    /// Returns the multiplication of `a` and `b`.
    #[inline(always)]
    fn mul(a: &u64, b: &u64) -> u64 {
        Self::from_u128(u128::from(*a) * u128::from(*b))
    }

    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        Self::weak_reduce(a + MERSENNE_61_PRIME_FIELD_ORDER - b)
    }

    /// Returns the additive inverse of `a`.
    #[inline(always)]
    fn neg(a: &u64) -> u64 {
        MERSENNE_61_PRIME_FIELD_ORDER - a
    }

    /// Returns the multiplicative inverse of `a`.
    /// Uses Fermat's little theorem: a^(-1) = a^(p-2) mod p
    /// p - 2 = 2^61 - 3
    #[inline]
    fn inv(x: &u64) -> Result<u64, FieldError> {
        if *x == Self::zero() || *x == MERSENNE_61_PRIME_FIELD_ORDER {
            return Err(FieldError::InvZeroError);
        }
        // Addition chain for x^(p-2) mod p.
        let p11 = Self::mul(&Self::square(x), x);
        let p101 = Self::mul(&Self::pow_2(x, 2), x);
        let p1111 = Self::mul(&Self::square(&p101), &p101);

        let p11110000 = Self::pow_2(&p1111, 4u64);
        let p11111111 = Self::mul(&p11110000, &p1111);

        let p1111111100 = Self::pow_2(&p11111111, 2u64);
        let p1111111111 = Self::mul(&p1111111100, &p11);
        let p1111111100000000 = &Self::pow_2(&p11111111, 8u64);
        let p1111111111111111 = &Self::mul(&p1111111100000000, &p11111111);

        let p11111111111111110000000000000000 = Self::pow_2(&p1111111111111111, 16u64);
        let p11111111111111111111111111111111 =
            Self::mul(&p11111111111111110000000000000000, &p1111111111111111);
        let p11111111111111110000000000 = Self::pow_2(&p1111111111111111, 10u64);
        let p11111111111111111111111111 = Self::mul(&p11111111111111110000000000, &p1111111111);

        let p1111111111111111111111111111111100000000000000000000000000 =
            Self::pow_2(&p11111111111111111111111111111111, 26u64);
        let p1111111111111111111111111111111111111111111111111111111111 = Self::mul(
            &p1111111111111111111111111111111100000000000000000000000000,
            &p11111111111111111111111111,
        );

        let p1111111111111111111111111111111111111111111111111111111111000 = Self::pow_2(
            &p1111111111111111111111111111111111111111111111111111111111,
            3u64,
        );
        let p1111111111111111111111111111111111111111111111111111111111101 = Self::mul(
            &p1111111111111111111111111111111111111111111111111111111111000,
            &p101,
        );
        Ok(p1111111111111111111111111111111111111111111111111111111111101)
    }

    /// Returns the division of `a` and `b`.
    #[inline]
    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(Self::mul(a, &b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    #[inline(always)]
    fn eq(a: &u64, b: &u64) -> bool {
        Self::as_representative(a) == Self::as_representative(b)
    }

    /// Returns the additive neutral element.
    #[inline(always)]
    fn zero() -> u64 {
        0u64
    }

    /// Returns the multiplicative neutral element.
    #[inline(always)]
    fn one() -> u64 {
        1u64
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    #[inline(always)]
    fn from_u64(x: u64) -> u64 {
        // Reduce a u64 modulo 2^61 - 1
        ((((((x + 1) >> 61) + x + 1) >> 61) + x) & (MERSENNE_61_PRIME_FIELD_ORDER)) as u64
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    #[inline(always)]
    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }

    #[inline(always)]
    fn double(a: &u64) -> u64 {
        Self::weak_reduce(a << 1)
    }
}

impl IsPrimeField for Mersenne61Field {
    type RepresentativeType = u64;

    fn representative(x: &u64) -> u64 {
        debug_assert!((x >> 61) == 0);
        Self::as_representative(x)
    }

    fn field_bit_size() -> usize {
        ((MERSENNE_61_PRIME_FIELD_ORDER - 1).ilog2() + 1) as usize
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let mut hex_string = hex_string;
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }
        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    fn to_hex(x: &u64) -> String {
        format!("{x:X}")
    }
}

impl Deref for FE1 {
    type Target = FieldElement<Mersenne61Field>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
