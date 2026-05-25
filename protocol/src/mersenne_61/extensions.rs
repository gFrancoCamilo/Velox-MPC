// Ported verbatim (sans tests) from async_mpc/fields/src/mersenne_61/extensions.rs.
// Degree-2 and degree-4 extension towers over Mersenne-61.

use super::field::Mersenne61Field;
use lambdaworks_math::field::{
    element::FieldElement,
    errors::FieldError,
    traits::{IsField, IsSubFieldOf},
};
use rand::random;

type FpE = FieldElement<Mersenne61Field>;
pub type Fp2E = FieldElement<Mersenne61Degree2ExtensionField>;
pub type Fp4E = FieldElement<Mersenne61Degree4ExtensionField>;

/// Degree-2 extension of the Mersenne-61 prime field.
/// Irreducible polynomial: x^2 + 1 = 0, so i^2 = -1.
/// Elements are represented as [a, b] = a + b*i.
#[derive(Clone, Debug)]
pub struct Mersenne61Degree2ExtensionField;

impl Mersenne61Degree2ExtensionField {
    /// Multiplies an Fp2 element by the non-residue (4 + i).
    /// (a + bi)(4 + i) = (4a - b) + (a + 4b)i
    pub fn mul_fp2_by_nonresidue(a: &Fp2E) -> Fp2E {
        let four = FpE::from(4u64);
        Fp2E::new([
            four * a.value()[0] - a.value()[1],
            a.value()[0] + four * a.value()[1],
        ])
    }
}

impl IsField for Mersenne61Degree2ExtensionField {
    type BaseType = [FpE; 2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// (a0 + a1*i)(b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
    /// Uses Karatsuba: imag = (a0+a1)*(b0+b1) - a0*b0 - a1*b1
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 - a1b1, z - a0b0 - a1b1]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 - a1);
        let c1 = v0.double();
        [c0, c1]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// (a0 + a1*i)^(-1) = (a0 - a1*i) / (a0^2 + a1^2)
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].square() + a[1].square()).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Mersenne61Degree2ExtensionField> for Mersenne61Field {
    fn add(
        a: &Self::BaseType,
        b: &<Mersenne61Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) + b[0], b[1]]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Mersenne61Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) - b[0], -b[1]]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Mersenne61Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) * b[0], FpE::from(a) * b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Mersenne61Degree2ExtensionField as IsField>::BaseType,
    ) -> Result<<Mersenne61Degree2ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Mersenne61Degree2ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Mersenne61Degree2ExtensionField>>::mul(a, &b_inv))
    }

    fn embed(a: Self::BaseType) -> <Mersenne61Degree2ExtensionField as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    fn to_subfield_vec(
        b: <Mersenne61Degree2ExtensionField as IsField>::BaseType,
    ) -> Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Degree-4 extension via tower construction: Fp2[w] / (w^2 - (4+i)),
/// where Fp2 = Fp[i] / (i^2 + 1). Elements are [a, b] representing a + b*w.
#[derive(Clone, Debug)]
pub struct Mersenne61Degree4ExtensionField;

impl Mersenne61Degree4ExtensionField {
    pub const fn const_from_coefficients(coeffs: [u64; 4]) -> Fp4E {
        Fp4E::const_from_raw([
            Fp2E::const_from_raw([
                FpE::const_from_raw(coeffs[0]),
                FpE::const_from_raw(coeffs[1]),
            ]),
            Fp2E::const_from_raw([
                FpE::const_from_raw(coeffs[2]),
                FpE::const_from_raw(coeffs[3]),
            ]),
        ])
    }

    pub fn rand_fe() -> Fp4E {
        Fp4E::new([
            Fp2E::new([FpE::new(random::<u64>()), FpE::new(random::<u64>())]),
            Fp2E::new([FpE::new(random::<u64>()), FpE::new(random::<u64>())]),
        ])
    }

    pub fn const_from_fe(elems: &[FpE]) -> Fp4E {
        Fp4E::const_from_raw([
            Fp2E::const_from_raw([elems[0], elems[1]]),
            Fp2E::const_from_raw([elems[2], elems[3]]),
        ])
    }
}

impl IsField for Mersenne61Degree4ExtensionField {
    type BaseType = [Fp2E; 2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    /// (a0 + a1*w)(b0 + b1*w) = (a0*b0 + a1*b1*(4+i)) + (a0*b1 + a1*b0)*w
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        [
            &a0b0 + Mersenne61Degree2ExtensionField::mul_fp2_by_nonresidue(&a1b1),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - a0b0 - a1b1,
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_square = &a[0].square();
        let a1_square = &a[1].square();
        [
            a0_square + Mersenne61Degree2ExtensionField::mul_fp2_by_nonresidue(a1_square),
            (&a[0] + &a[1]).square() - a0_square - a1_square,
        ]
    }

    /// norm = a0^2 - (4+i)*a1^2; inv = [a0/norm, -a1/norm].
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].square()
            - Mersenne61Degree2ExtensionField::mul_fp2_by_nonresidue(&a[1].square()))
        .inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * &inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [Fp2E::zero(), Fp2E::zero()]
    }

    fn one() -> Self::BaseType {
        [Fp2E::one(), Fp2E::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [Fp2E::from(x), Fp2E::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Mersenne61Degree4ExtensionField> for Mersenne61Field {
    fn add(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [FpE::from(a) + &b[0], b[1].clone()]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [FpE::from(a) - &b[0], -&b[1]]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [FpE::from(a) * &b[0], FpE::from(a) * &b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> Result<<Mersenne61Degree4ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Mersenne61Degree4ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Mersenne61Degree4ExtensionField>>::mul(a, &b_inv))
    }

    fn embed(a: Self::BaseType) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [
            Fp2E::from_raw(<Self as IsSubFieldOf<Mersenne61Degree2ExtensionField>>::embed(a)),
            Fp2E::zero(),
        ]
    }

    fn to_subfield_vec(
        b: <Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> Vec<Self::BaseType> {
        b.into_iter()
            .flat_map(|fp2e| [fp2e.value()[0].to_raw(), fp2e.value()[1].to_raw()])
            .collect()
    }
}

impl IsSubFieldOf<Mersenne61Degree4ExtensionField> for Mersenne61Degree2ExtensionField {
    fn mul(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        let a = Fp2E::from_raw(*a);
        [&a * &b[0], &a * &b[1]]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [Fp2E::from_raw(*a) + &b[0], b[1].clone()]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [Fp2E::from_raw(*a) - &b[0], -&b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> Result<<Mersenne61Degree4ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Mersenne61Degree4ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Mersenne61Degree4ExtensionField>>::mul(a, &b_inv))
    }

    fn embed(a: Self::BaseType) -> <Mersenne61Degree4ExtensionField as IsField>::BaseType {
        [Fp2E::from_raw(a), Fp2E::zero()]
    }

    fn to_subfield_vec(
        b: <Mersenne61Degree4ExtensionField as IsField>::BaseType,
    ) -> Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}
