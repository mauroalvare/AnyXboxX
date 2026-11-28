//! IEEE division helpers of RDNA 2 (VOP3): `v_div_scale_f32/f64`, `v_div_fmas_f32/f64` and
//! `v_div_fixup_f32/f64`. Port of `emu/rdna2_exec_div.py`.
//!
//! The compilers divide n / d in four steps: v_div_scale rescales d and n (and sets VCC when a correction
//! will be needed), reciprocal and Newton steps compute the quotient, v_div_fmas does the last fused
//! multiply-add and undoes the scale when VCC is set, and v_div_fixup handles the special cases.
//!
//! The model rounds exact rationals (Python's `Fraction`). The exact values are held as big integers
//! (`num-bigint`): numerator and denominator are not reduced, which does not change any result here (the
//! power estimate and the round-half-even test depend only on the value). Choices the Python documents as
//! assumptions (the FMA scale of v_div_fmas, the NaN of v_div_scale, the f64 branch that is not modelled)
//! are kept as the Python has them, and the cases it refuses are refused.

use num_bigint::{BigInt, Sign};

use crate::compare::read64;
use crate::error::Error;
use crate::vop3::Fields;
use crate::wave::Wave;

const LITERAL: u32 = 255;

/// One IEEE format of the division helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub width: u32,
    pub mantissa: u32,
    pub bias: i64,
    /// Exponent of the scale of v_div_scale and v_div_fmas.
    pub scale: i64,
    /// Exponent gap of the « numerator and denominator near the maximum » branch.
    pub near_max: i64,
    /// Biased exponent below which a numerator is « tiny ».
    pub tiny: i64,
    pub kind: &'static str,
}

pub const F32: Format = Format {
    width: 32,
    mantissa: 23,
    bias: 127,
    scale: 64,
    near_max: 96,
    tiny: 23,
    kind: "f32",
};
pub const F64: Format = Format {
    width: 64,
    mantissa: 52,
    bias: 1023,
    scale: 128,
    near_max: 768,
    tiny: 53,
    kind: "f64",
};

impl Format {
    fn max_exponent(self) -> u64 {
        (1u64 << (self.width - 1 - self.mantissa)) - 1
    }
    fn sign(self) -> u64 {
        1u64 << (self.width - 1)
    }
    fn quiet(self) -> u64 {
        1u64 << (self.mantissa - 1)
    }
    fn negative_nan(self) -> u64 {
        self.sign() | self.max_exponent() << self.mantissa | self.quiet()
    }
    fn nan(self) -> u64 {
        self.max_exponent() << self.mantissa | self.quiet()
    }
    fn infinity(self) -> u64 {
        self.max_exponent() << self.mantissa
    }
    fn mantissa_mask(self) -> u64 {
        (1u64 << self.mantissa) - 1
    }
    fn word_mask(self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }
}

/// An exact rational `num / den` with `den > 0` (not reduced).
#[derive(Debug, Clone)]
struct Frac {
    num: BigInt,
    den: BigInt,
}

fn is_zero(value: &BigInt) -> bool {
    value.sign() == Sign::NoSign
}

fn is_negative(value: &BigInt) -> bool {
    value.sign() == Sign::Minus
}

impl Frac {
    fn new(num: BigInt, den: BigInt) -> Frac {
        if is_negative(&den) {
            Frac {
                num: -num,
                den: -den,
            }
        } else {
            Frac { num, den }
        }
    }

    /// The exact value of a finite double.
    fn from_f64(x: f64) -> Frac {
        let bits = x.to_bits();
        let negative = bits >> 63 == 1;
        let exponent = ((bits >> 52) & 0x7FF) as i64;
        let mantissa = bits & ((1u64 << 52) - 1);
        let (significand, exp) = if exponent == 0 {
            (mantissa, -1074)
        } else {
            (mantissa | 1u64 << 52, exponent - 1075)
        };
        let mut num = BigInt::from(significand);
        let mut den = BigInt::from(1);
        if exp >= 0 {
            num <<= exp as usize;
        } else {
            den <<= (-exp) as usize;
        }
        if negative {
            num = -num;
        }
        Frac { num, den }
    }

    /// 2^e, for any integer e.
    fn pow2(e: i64) -> Frac {
        if e >= 0 {
            Frac {
                num: BigInt::from(1) << e as usize,
                den: BigInt::from(1),
            }
        } else {
            Frac {
                num: BigInt::from(1),
                den: BigInt::from(1) << (-e) as usize,
            }
        }
    }

    fn mul(&self, other: &Frac) -> Frac {
        Frac::new(&self.num * &other.num, &self.den * &other.den)
    }

    fn add(&self, other: &Frac) -> Frac {
        Frac::new(
            &self.num * &other.den + &other.num * &self.den,
            &self.den * &other.den,
        )
    }

    fn div(&self, other: &Frac) -> Frac {
        Frac::new(&self.num * &other.den, &self.den * &other.num)
    }

    fn abs(&self) -> Frac {
        if is_negative(&self.num) {
            Frac {
                num: -self.num.clone(),
                den: self.den.clone(),
            }
        } else {
            self.clone()
        }
    }

    /// self > other, both with positive denominators.
    fn greater(&self, other: &Frac) -> bool {
        &self.num * &other.den > &other.num * &self.den
    }
}

/// Bits of the exact value rounded to nearest-even in `fmt` (subnormals and infinities included).
fn round_exact(exact: &Frac, fmt: Format, negative_zero: bool) -> u64 {
    if is_zero(&exact.num) {
        return if negative_zero { fmt.sign() } else { 0 };
    }
    let sign = if is_negative(&exact.num) {
        fmt.sign()
    } else {
        0
    };
    let magnitude = exact.abs();
    let mut power = magnitude.num.bits() as i64 - magnitude.den.bits() as i64;
    if Frac::pow2(power).greater(&magnitude) {
        power -= 1;
    }
    power = power.max(1 - fmt.bias);
    let shift = i64::from(fmt.mantissa) - power;
    let scaled = magnitude.mul(&Frac::pow2(shift));
    let mut integer = &scaled.num / &scaled.den;
    let remainder = &scaled.num % &scaled.den;
    let twice = &remainder * 2;
    let odd = (&integer & BigInt::from(1)) == BigInt::from(1);
    if twice > scaled.den || (twice == scaled.den && odd) {
        integer += BigInt::from(1);
    }
    if is_nonzero(&(&integer >> (fmt.mantissa + 1) as usize)) {
        integer >>= 1usize;
        power += 1;
    }
    if power > fmt.bias {
        return sign | fmt.infinity();
    }
    let integer = u64::try_from(integer).unwrap_or(0);
    if integer >> fmt.mantissa == 0 {
        return sign | integer;
    }
    sign | ((power + fmt.bias) as u64) << fmt.mantissa | (integer - (1u64 << fmt.mantissa))
}

fn is_nonzero(value: &BigInt) -> bool {
    !is_zero(value)
}

fn value(bits: u64, fmt: Format) -> f64 {
    if fmt.width == 32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    }
}

fn exponent(bits: u64, fmt: Format) -> u64 {
    bits >> fmt.mantissa & fmt.max_exponent()
}

fn is_nan(bits: u64, fmt: Format) -> bool {
    exponent(bits, fmt) == fmt.max_exponent() && bits & fmt.mantissa_mask() != 0
}

fn is_denormal(bits: u64, fmt: Format) -> bool {
    exponent(bits, fmt) == 0 && bits & fmt.mantissa_mask() != 0
}

/// ldexp(S0, power): NaN and infinities unchanged, otherwise the exact value scaled and rounded.
fn scaled(bits: u64, power: i64, fmt: Format) -> u64 {
    if exponent(bits, fmt) == fmt.max_exponent() {
        return bits;
    }
    let number = Frac::from_f64(value(bits, fmt));
    round_exact(&number.mul(&Frac::pow2(power)), fmt, bits & fmt.sign() != 0)
}

/// Whether S2 / S1 rounds to a non-zero denormal (both finite and non-zero).
fn quotient_is_denormal(numerator: u64, denominator: u64, fmt: Format) -> bool {
    let quotient =
        Frac::from_f64(value(numerator, fmt)).div(&Frac::from_f64(value(denominator, fmt)));
    let bits = round_exact(&quotient, fmt, false);
    is_denormal(bits & !fmt.sign(), fmt)
}

/// (result, VCC bit) of v_div_scale: S0 to rescale, S1 the denominator, S2 the numerator.
pub fn div_scale(s0: u64, s1: u64, s2: u64, fmt: Format) -> Result<(u64, bool), Error> {
    let (x0, x1, x2) = (value(s0, fmt), value(s1, fmt), value(s2, fmt));
    if x2 == 0.0 || x1 == 0.0 {
        return Ok((fmt.nan(), false));
    }
    if exponent(s2, fmt) as i64 - exponent(s1, fmt) as i64 >= fmt.near_max {
        let result = if x0 == x1 {
            scaled(s0, fmt.scale, fmt)
        } else {
            s0
        };
        return Ok((result, true));
    }
    if is_denormal(s1, fmt) {
        return Ok((scaled(s0, fmt.scale, fmt), false));
    }
    if fmt.width == 64 && x1.is_finite() && x1.abs() > 2f64.powi(1022) {
        return Err(Error::Untranslated(
            "v_div_scale_f64 : 1/S1 dénormal, pseudo-code d'AMD incohérent".into(),
        ));
    }
    let finite = x1.is_finite() && x2.is_finite();
    if finite && quotient_is_denormal(s2, s1, fmt) {
        let result = if x0 == x2 {
            scaled(s0, fmt.scale, fmt)
        } else {
            s0
        };
        return Ok((result, true));
    }
    if exponent(s2, fmt) as i64 <= fmt.tiny {
        return Ok((scaled(s0, fmt.scale, fmt), false));
    }
    Ok((s0, false))
}

/// v_div_fmas: S0 × S1 + S2 rounded once, scaled by 2^±scale when VCC is set (sign by |S2| ≥ 2).
pub fn div_fmas(s0: u64, s1: u64, s2: u64, vcc: bool, fmt: Format) -> u64 {
    let (x0, x1, x2) = (value(s0, fmt), value(s1, fmt), value(s2, fmt));
    let power = if vcc {
        if x2.abs() >= 2.0 {
            fmt.scale
        } else {
            -fmt.scale
        }
    } else {
        0
    };
    if ![x0, x1, x2].iter().all(|x| x.is_finite()) {
        let special = x0 * x1 + x2;
        return if special.is_nan() {
            fmt.nan()
        } else {
            (if special < 0.0 { fmt.sign() } else { 0 }) | fmt.infinity()
        };
    }
    let exact = Frac::from_f64(x0)
        .mul(&Frac::from_f64(x1))
        .add(&Frac::from_f64(x2));
    let negative_zero = (x0 * x1 + x2).is_sign_negative();
    round_exact(&exact.mul(&Frac::pow2(power)), fmt, negative_zero)
}

/// v_div_fixup: the quotient S0 corrected for the special denominator S1 and numerator S2.
pub fn div_fixup(s0: u64, s1: u64, s2: u64, fmt: Format) -> u64 {
    let sign = (s1 ^ s2) & fmt.sign();
    let magnitude = !fmt.sign() & fmt.word_mask();
    let (x1, x2) = (value(s1, fmt), value(s2, fmt));
    if is_nan(s2, fmt) {
        return s2 | fmt.quiet();
    }
    if is_nan(s1, fmt) {
        return s1 | fmt.quiet();
    }
    if (x1 == 0.0 && x2 == 0.0) || (x1.is_infinite() && x2.is_infinite()) {
        return fmt.negative_nan();
    }
    if x1 == 0.0 || x2.is_infinite() {
        return sign | fmt.infinity();
    }
    if x1.is_infinite() || x2 == 0.0 {
        return sign;
    }
    let gap = exponent(s2, fmt) as i64 - exponent(s1, fmt) as i64;
    if gap < -(fmt.bias + i64::from(fmt.mantissa)) {
        return sign;
    }
    if is_nan(s0, fmt) {
        return sign | fmt.infinity();
    }
    sign | (s0 & magnitude)
}

/// `_modify` of the model for f32 and f64 operands: abs, then neg, on the sign bit.
fn modify(bits: u64, fmt: Format, index: u32, abs_bits: u32, neg_bits: u32) -> u64 {
    let sign = fmt.sign();
    let mut out = bits;
    if abs_bits >> index & 1 == 1 {
        out &= !sign;
    }
    if neg_bits >> index & 1 == 1 {
        out ^= sign;
    }
    out
}

/// The division helper of `name`, if it is one.
pub fn operation(name: &str) -> Option<Format> {
    match name {
        "v_div_scale_f32" | "v_div_fmas_f32" | "v_div_fixup_f32" => Some(F32),
        "v_div_scale_f64" | "v_div_fmas_f64" | "v_div_fixup_f64" => Some(F64),
        _ => None,
    }
}

/// Executes a division helper for every active lane. `Ok(false)` when `name` is not one of them.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let Some(fmt) = operation(name) else {
        return Ok(false);
    };
    if fields.clamp || fields.omod != 0 {
        return Err(Error::Untranslated(format!("{name} avec clamp/omod")));
    }
    let is_scale = name.starts_with("v_div_scale");
    let abs_bits = if is_scale { 0 } else { fields.abs };
    let vcc = wave.vcc()?;
    let mut vcc_mask = 0u64;
    let destination = fields.vdst as usize;
    for lane in wave.active()? {
        let mut operands = [0u64; 3];
        for (index, code) in fields.sources.iter().enumerate() {
            let raw = if fmt.width == 32 {
                u64::from(wave.read_lane(*code, lane, literal)?)
            } else {
                if *code == LITERAL {
                    return Err(Error::Untranslated(
                        "littéral sur une source 64 bits".into(),
                    ));
                }
                read64(wave, *code, lane, true)?
            };
            operands[index] = modify(raw, fmt, index as u32, abs_bits, fields.neg);
        }
        let [s0, s1, s2] = operands;
        let result = if is_scale {
            let (result, bit) = div_scale(s0, s1, s2, fmt)?;
            vcc_mask |= u64::from(bit) << lane;
            result
        } else if name.starts_with("v_div_fmas") {
            div_fmas(s0, s1, s2, vcc >> lane & 1 == 1, fmt)
        } else {
            div_fixup(s0, s1, s2, fmt)
        };
        wave.vgpr_group(destination, if fmt.width == 64 { 2 } else { 1 })?;
        wave.v[destination][lane] = (result & 0xFFFF_FFFF) as u32;
        if fmt.width == 64 {
            wave.v[destination + 1][lane] = (result >> 32) as u32;
        }
    }
    if is_scale {
        wave.write_scalar(fields.sdst, vcc_mask, 1)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rounding_matches_ieee_for_simple_values() {
        // 1 / 3 rounded to f32 is 0x3EAAAAAB.
        let third = Frac::from_f64(1.0).div(&Frac::from_f64(3.0));
        assert_eq!(round_exact(&third, F32, false), 0x3EAA_AAAB);
        assert_eq!(round_exact(&Frac::from_f64(0.0), F32, true), F32.sign());
        // 2^128 overflows f32 to infinity.
        assert_eq!(round_exact(&Frac::pow2(128), F32, false), F32.infinity());
    }

    #[test]
    fn fixup_of_zero_over_zero_is_the_negative_nan() {
        let zero = 0u64;
        assert_eq!(div_fixup(0x3F80_0000, zero, zero, F32), F32.negative_nan());
    }

    #[test]
    fn from_f64_is_exact_up_to_a_common_factor() {
        let half = Frac::from_f64(0.5);
        assert_eq!(half.num.clone() * BigInt::from(2), half.den.clone());
    }
}
