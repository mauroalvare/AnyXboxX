//! binary16 (f16) values of the model (`emu/rdna2_exec_f16.py`: `half_bits`, `bits_half`, `_is_nan`, `half_min`,
//! `half_max`, `half_clamp`, `INLINE_HALVES`). Arithmetic is done in double precision, then the result is rounded
//! once to binary16 with ties to even, as Python's `struct.pack("<e")` does. Any NaN result is `0x7E00`.

use crate::error::Error;
use crate::wave::Wave;

pub const HALF_NAN: u32 = 0x7E00;
pub const HALF_INFINITY: u32 = 0x7C00;
pub const HALF_ONE: u32 = 0x3C00;
pub const ALMOST_ONE_HALF: u32 = 0x3BFF;
pub const HALF_MASK: u32 = 0xFFFF;
pub const HALF_SIGN: u32 = 0x8000;
/// `FLOAT_NAN` of the model: the f32 NaN given to a NaN half converted to single precision.
pub const FLOAT_NAN: u32 = 0x7FC0_0000;

/// Source operand kind of scalar f16, f16 compare and f16 conversion instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Half,
    Word,
    Int16,
}

/// Inline float constants 240 to 248 read by a half-precision source (`INLINE_HALVES`).
const INLINE_HALVES: [(u32, u32); 9] = [
    (240, 0x3800),
    (241, 0xB800),
    (242, 0x3C00),
    (243, 0xBC00),
    (244, 0x4000),
    (245, 0xC000),
    (246, 0x4400),
    (247, 0xC400),
    (248, 0x3118),
];

/// 2^k as a double, exact for k in -1022..=1023.
pub fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The value of binary16 bits (NaN for every NaN encoding).
pub fn half_to_f64(bits: u32) -> f64 {
    let sign = if bits & HALF_SIGN != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = f64::from(bits & 0x3FF);
    let magnitude = match exponent {
        0x1F if mantissa == 0.0 => f64::INFINITY,
        0x1F => f64::NAN,
        0 => mantissa * pow2(-24),
        _ => (1024.0 + mantissa) * pow2(exponent as i32 - 25),
    };
    sign * magnitude
}

/// Binary16 bits of a double, rounded once to nearest with ties to even. NaN gives `HALF_NAN`; a value
/// past the largest finite half (65504, rounding included) gives ±infinity.
pub fn half_from_f64(value: f64) -> u32 {
    if value.is_nan() {
        return HALF_NAN;
    }
    let sign = if value.is_sign_negative() {
        HALF_SIGN
    } else {
        0
    };
    let magnitude = value.abs();
    if magnitude.is_infinite() {
        return sign | HALF_INFINITY;
    }
    if magnitude == 0.0 {
        return sign;
    }
    // Subnormal halves share the quantum of the smallest normal exponent (2^-14, step 2^-24).
    let exponent = ((magnitude.to_bits() >> 52) as i32 - 1023).max(-14);
    let steps = (magnitude * pow2(10 - exponent)).round_ties_even() as u32;
    let bits = if steps < 1024 {
        steps
    } else {
        (((exponent + 15) as u32) << 10) + steps - 1024
    };
    sign | bits.min(HALF_INFINITY)
}

/// `_is_nan`: an exponent of all ones with a non-zero mantissa.
pub fn is_nan(bits: u32) -> bool {
    bits & 0x7FFF > HALF_INFINITY
}

/// The f32 bits of a half (exact), with the model's NaN.
pub fn half_to_f32_bits(bits: u32) -> u32 {
    if is_nan(bits) {
        FLOAT_NAN
    } else {
        (half_to_f64(bits) as f32).to_bits()
    }
}

/// `_order`: a key that grows with the value (NaN excluded), -0 just below +0.
fn order(bits: u32) -> u32 {
    if bits & HALF_SIGN != 0 {
        !bits & HALF_MASK
    } else {
        bits | HALF_SIGN
    }
}

/// `half_min`: a NaN gives way to the other operand, -0 < +0.
pub fn half_min(a: u32, b: u32) -> u32 {
    if is_nan(a) {
        return if is_nan(b) { HALF_NAN } else { b };
    }
    if is_nan(b) {
        return a;
    }
    if order(a) <= order(b) { a } else { b }
}

/// `half_max`: a NaN gives way to the other operand.
pub fn half_max(a: u32, b: u32) -> u32 {
    if is_nan(a) {
        return if is_nan(b) { HALF_NAN } else { b };
    }
    if is_nan(b) {
        return a;
    }
    if order(a) >= order(b) { a } else { b }
}

/// `half_clamp`: the VOP3 clamp of a half result to [0, 1]; NaN and negatives (-0 included) give +0.
pub fn half_clamp(bits: u32) -> u32 {
    if is_nan(bits) || bits & HALF_SIGN != 0 {
        0
    } else {
        bits.min(HALF_ONE)
    }
}

/// The inline half of a source code (240 to 248), if it is one.
pub fn inline_half(code: u32) -> Option<u32> {
    INLINE_HALVES
        .iter()
        .find(|(inline, _)| *inline == code)
        .map(|(_, bits)| *bits)
}

/// `half_toward_zero`: binary16 bits of `value` rounded toward zero; finite values past ±65504 give ±65504.
pub fn half_toward_zero(value: f64) -> u32 {
    if value.is_nan() || value.is_infinite() {
        return half_from_f64(value);
    }
    let bits = half_from_f64(value);
    if half_to_f64(bits).abs() > value.abs() {
        bits - 1
    } else {
        bits
    }
}

/// `half_round`: IEEE roundToIntegral in binary16 keeping the sign (`trunc(-0.5) = -0`), ±inf unchanged, NaN -> `0x7E00`.
pub fn half_round(bits: u32, rounding: fn(f64) -> f64) -> u32 {
    if is_nan(bits) {
        return HALF_NAN;
    }
    if bits & 0x7FFF == HALF_INFINITY {
        return bits;
    }
    let value = half_to_f64(bits);
    half_from_f64(rounding(value).copysign(value))
}

/// `half_fract`: `S0 - floor(S0)` clamped at `0x3BFF`; ±infinity and NaN give `0x7E00`.
pub fn half_fract(bits: u32) -> u32 {
    if is_nan(bits) || bits & 0x7FFF == HALF_INFINITY {
        return HALF_NAN;
    }
    let value = half_to_f64(bits);
    let frac = value - value.floor().copysign(value);
    half_from_f64(frac.min(half_to_f64(ALMOST_ONE_HALF)))
}

/// `half_frexp`: `(mantissa_f16, exponent)` of `S0 = m * 2^e` with `0.5 <= |m| < 1`; ±0, ±inf and NaN give `(bits, 0)`.
pub fn half_frexp(bits: u32) -> (u32, i32) {
    if bits & 0x7FFF == 0 || bits & 0x7FFF >= HALF_INFINITY {
        return (bits, 0);
    }
    let (mantissa, exponent) = crate::float::frexp(half_to_f64(bits));
    (half_from_f64(mantissa), exponent)
}

/// `_source` of `emu/rdna2_exec_f16.py`: reads a 32-bit word (`Word`) or a 16-bit half (`Half` / `Int16`).
pub fn read_source(
    wave: &Wave,
    name: &str,
    kind: SourceKind,
    code: u32,
    lane: usize,
    literal: Option<u32>,
    high: bool,
) -> Result<u32, Error> {
    if kind == SourceKind::Word {
        return wave.read_lane(code, lane, literal);
    }
    if !(128..crate::wave::VGPR_FIRST).contains(&code) {
        let word = wave.read_lane(code, lane, literal)?;
        return Ok(if high { word >> 16 } else { word & HALF_MASK });
    }
    if high {
        return Err(Error::Untranslated(format!(
            "{name}: op_sel on the high half of a constant not supported"
        )));
    }
    if let Some(bits) = inline_half(code) {
        if kind == SourceKind::Int16 {
            return Err(Error::Untranslated(format!(
                "{name}: float inline constant on an integer source not supported"
            )));
        }
        return Ok(bits);
    }
    Ok((wave.read_scalar(code, literal, 1)? as u32) & HALF_MASK)
}

/// `_modified` of `emu/rdna2_exec_f16.py`: VOP3 `abs` and `neg` on bit 15 (`Half`) or bit 31 (`Word`).
pub fn modify_source(mut value: u32, kind: SourceKind, abs: u32, neg: u32, index: usize) -> u32 {
    if kind == SourceKind::Int16 {
        return value;
    }
    let sign = if kind == SourceKind::Word {
        0x8000_0000
    } else {
        HALF_SIGN
    };
    if (abs >> index) & 1 != 0 {
        value &= !sign;
    }
    if (neg >> index) & 1 != 0 {
        value ^= sign;
    }
    value
}

/// `_read_pk_half`: one 16-bit half of a source. Inline constants repeat in both halves; a register or
/// literal word gives its high half when `high`.
pub fn read_half(
    wave: &Wave,
    code: u32,
    lane: usize,
    literal: Option<u32>,
    high: bool,
) -> Result<u32, Error> {
    if let Some(bits) = inline_half(code) {
        return Ok(bits);
    }
    let word = wave.read_lane(code, lane, literal)?;
    Ok(if high { word >> 16 } else { word & HALF_MASK })
}

/// `read_half` with the negation of the source applied (`^ HALF_SIGN`).
pub fn signed_half(
    wave: &Wave,
    code: u32,
    lane: usize,
    literal: Option<u32>,
    high: bool,
    negate: bool,
) -> Result<u32, Error> {
    let bits = read_half(wave, code, lane, literal, high)?;
    Ok(if negate { bits ^ HALF_SIGN } else { bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_once_to_nearest_even() {
        assert_eq!(half_from_f64(1.0), 0x3C00);
        assert_eq!(half_from_f64(65504.0), 0x7BFF);
        assert_eq!(half_from_f64(65519.0), 0x7BFF);
        assert_eq!(half_from_f64(65520.0), HALF_INFINITY);
        assert_eq!(half_from_f64(-70000.0), 0xFC00);
        // 2^-25 is a tie between 0 and the smallest subnormal: ties to even gives 0.
        assert_eq!(half_from_f64(pow2(-25)), 0);
        // 3 × 2^-25 is a tie between 1 and 2 units of 2^-24: ties to even gives 2.
        assert_eq!(half_from_f64(3.0 * pow2(-25)), 2);
        assert_eq!(half_from_f64(pow2(-14)), 0x0400);
        // One rounding only: 1 + 2^-11 + 2^-40 is above the tie, so it goes up (double rounding gives 1.0).
        assert_eq!(half_from_f64(1.0 + pow2(-11) + pow2(-40)), 0x3C01);
        assert_eq!(half_from_f64(-0.0), HALF_SIGN);
        assert_eq!(half_from_f64(f64::NAN), HALF_NAN);
        assert_eq!(half_from_f64(f64::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn every_finite_half_round_trips() {
        for bits in 0..=0xFFFFu32 {
            if is_nan(bits) {
                continue;
            }
            assert_eq!(half_from_f64(half_to_f64(bits)), bits, "bits 0x{bits:04X}");
        }
    }

    #[test]
    fn min_max_and_clamp_follow_the_model() {
        assert_eq!(half_min(0x8000, 0x0000), 0x8000);
        assert_eq!(half_max(0x8000, 0x0000), 0x0000);
        assert_eq!(half_min(HALF_NAN, 0x3C00), 0x3C00);
        assert_eq!(half_max(0x3C00, HALF_NAN), 0x3C00);
        assert_eq!(half_clamp(0x4400), HALF_ONE);
        assert_eq!(half_clamp(0x8000), 0);
        assert_eq!(half_clamp(0xBC00), 0);
        assert_eq!(half_clamp(HALF_NAN), 0);
    }
}
