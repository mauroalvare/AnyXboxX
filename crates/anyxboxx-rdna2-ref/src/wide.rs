//! Instructions with 64-bit operands (double-precision floats, 64-bit shifts): port of
//! `emu/rdna2_exec_wide.py`. A 64-bit source is a VGPR or SGPR pair, or an inline constant; a literal is
//! refused. VOP3 input modifiers act on the sign of float sources; clamp and omod on float results.
//!
//! v_rcp_f64, v_rsq_f64 and v_sqrt_f64 are rounded to nearest here; the ISA allows a few ulp more, as the
//! Python notes.

use crate::compare::read64;
use crate::error::Error;
use crate::float::{CANONICAL_NAN64, bits_float, float_bits, ldexp, py_max, py_min};
use crate::scalar::sign_extend;
use crate::wave::Wave;

const MASK32: u64 = 0xFFFF_FFFF;
const QUIET_F64: u64 = 1 << 51;
const ALMOST_ONE_F64: u64 = 0x3FEF_FFFF_FFFF_FFFF;
const SIGN32: u64 = 1 << 31;
const SIGN64: u64 = 1 << 63;

/// The kind of an operand or result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    F32,
    F64,
    I32,
    U32,
    U64,
}

impl Kind {
    /// Operands and results of 64 bits: a register pair.
    fn is_wide(self) -> bool {
        matches!(self, Kind::F64 | Kind::U64)
    }

    /// The float kinds: input modifiers and output modifiers apply.
    fn is_float(self) -> bool {
        matches!(self, Kind::F32 | Kind::F64)
    }

    fn sign(self) -> u64 {
        if self == Kind::F32 { SIGN32 } else { SIGN64 }
    }
}

/// The source kinds and result kind of a 64-bit instruction (the Python `OPERATIONS` table).
pub fn spec(name: &str) -> Option<(&'static [Kind], Kind)> {
    use Kind::{F32, F64, I32, U32, U64};
    Some(match name {
        "v_cvt_i32_f64" => (&[F64], I32),
        "v_cvt_u32_f64" => (&[F64], U32),
        "v_cvt_f64_i32" => (&[I32], F64),
        "v_cvt_f64_u32" => (&[U32], F64),
        "v_cvt_f32_f64" => (&[F64], F32),
        "v_cvt_f64_f32" => (&[F32], F64),
        "v_trunc_f64" | "v_ceil_f64" | "v_floor_f64" | "v_rndne_f64" | "v_fract_f64"
        | "v_rcp_f64" | "v_rsq_f64" | "v_sqrt_f64" | "v_frexp_mant_f64" => (&[F64], F64),
        "v_frexp_exp_i32_f64" => (&[F64], I32),
        "v_add_f64" | "v_mul_f64" | "v_min_f64" | "v_max_f64" => (&[F64, F64], F64),
        "v_fma_f64" => (&[F64, F64, F64], F64),
        "v_ldexp_f64" => (&[F64, I32], F64),
        "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => (&[U32, U64], U64),
        _ => return None,
    })
}

/// `_rounded`: an infinity stays, a NaN is quieted, otherwise the rounding with the sign of the input.
fn rounded(x: f64, rounding: fn(f64) -> f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | QUIET_F64);
    }
    if x.is_infinite() {
        x
    } else {
        rounding(x).copysign(x)
    }
}

/// `_to_int` for a 64-bit float to a 32-bit integer range.
fn to_int(value: f64, low: f64, high: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value > 0.0 { high as i64 } else { low as i64 }
    } else {
        value.trunc().clamp(low, high) as i64
    }
}

/// `_fract`: x + -floor(x), at most the largest double below 1; NaN quieted; an infinity gives NaN.
fn fract(x: f64) -> f64 {
    if x.is_nan() {
        return f64::from_bits(x.to_bits() | QUIET_F64);
    }
    if x.is_infinite() {
        return f64::NAN;
    }
    py_min(x - rounded(x, f64::floor), f64::from_bits(ALMOST_ONE_F64))
}

/// `_pick`: min and max of two doubles where one NaN is ignored (minNum and maxNum).
fn pick(x: f64, y: f64, x_wins: fn(f64, f64) -> bool) -> f64 {
    if x.is_nan() || y.is_nan() {
        return if x.is_nan() { y } else { x };
    }
    if x_wins(x, y) { x } else { y }
}

/// `_output` of the model: omod (×2, ×4, ÷2) then clamp to [0, 1], for float results.
fn output(value: f64, clamp: bool, omod: u32) -> f64 {
    if value.is_nan() || !(clamp || omod != 0) {
        return value;
    }
    let scaled = value
        * match omod {
            1 => 2.0,
            2 => 4.0,
            3 => 0.5,
            _ => 1.0,
        };
    if clamp {
        py_min(py_max(scaled, 0.0), 1.0)
    } else {
        scaled
    }
}

/// The result bits of one lane: the operation on decoded operands, output modifiers, then the encoding.
fn compute(name: &str, values: &[u64], clamp: bool, omod: u32) -> Option<u64> {
    let f = |index: usize| f64::from_bits(values[index]);
    let g = |index: usize| bits_float(values[index] as u32);
    let i = |index: usize| sign_extend(values[index], 32);
    let float_result = |value: f64| {
        let value = output(value, clamp, omod);
        value.to_bits()
    };
    let result: u64 = match name {
        "v_cvt_i32_f64" => (to_int(f(0), f64::from(i32::MIN), f64::from(i32::MAX)) as u64) & MASK32,
        "v_cvt_u32_f64" => (to_int(f(0), 0.0, MASK32 as f64) as u64) & MASK32,
        "v_cvt_f64_i32" => float_result(i(0) as f64),
        "v_cvt_f64_u32" => float_result(values[0] as f64),
        "v_cvt_f32_f64" => float_bits(output(f(0), clamp, omod)) as u64,
        "v_cvt_f64_f32" => float_result(g(0)),
        "v_trunc_f64" => float_result(rounded(f(0), f64::trunc)),
        "v_ceil_f64" => float_result(rounded(f(0), f64::ceil)),
        "v_floor_f64" => float_result(rounded(f(0), f64::floor)),
        "v_rndne_f64" => float_result(rounded(f(0), f64::round_ties_even)),
        "v_fract_f64" => float_result(fract(f(0))),
        "v_rcp_f64" => {
            let x = f(0);
            float_result(if x == 0.0 {
                f64::INFINITY.copysign(x)
            } else {
                1.0 / x
            })
        }
        // A NaN result is the bit pattern itself: LLVM may fold a NaN value through a select and
        // change its sign, so the canonical bits are not passed through f64.
        "v_rsq_f64" => {
            let x = f(0);
            if x < 0.0 {
                CANONICAL_NAN64
            } else {
                float_result(if x == 0.0 {
                    f64::INFINITY.copysign(x)
                } else {
                    1.0 / x.sqrt()
                })
            }
        }
        "v_sqrt_f64" => {
            let x = f(0);
            if x < 0.0 {
                CANONICAL_NAN64
            } else {
                float_result(x.sqrt())
            }
        }
        "v_frexp_exp_i32_f64" => {
            let x = f(0);
            (crate::float::frexp(x).1 as i64 as u64) & MASK32
        }
        "v_frexp_mant_f64" => float_result(crate::float::frexp(f(0)).0),
        "v_add_f64" => float_result(f(0) + f(1)),
        "v_mul_f64" => float_result(f(0) * f(1)),
        "v_min_f64" => float_result(pick(f(0), f(1), |x, y| x < y)),
        "v_max_f64" => float_result(pick(f(0), f(1), |x, y| x > y)),
        "v_fma_f64" => float_result(f(0).mul_add(f(1), f(2))),
        "v_ldexp_f64" => float_result(ldexp(f(0), i(1))),
        "v_lshlrev_b64" => values[1] << (values[0] & 63),
        "v_lshrrev_b64" => values[1] >> (values[0] & 63),
        "v_ashrrev_i64" => ((values[1] as i64) >> (values[0] & 63)) as u64,
        _ => return None,
    };
    Some(result)
}

/// `_modify`: abs then neg on the sign bit of one float operand.
fn modify(bits: u64, kind: Kind, index: u32, abs_bits: u32, neg_bits: u32) -> u64 {
    if !kind.is_float() {
        return bits;
    }
    let mut out = bits;
    if abs_bits >> index & 1 == 1 {
        out &= !kind.sign();
    }
    if neg_bits >> index & 1 == 1 {
        out ^= kind.sign();
    }
    out
}

/// Executes a 64-bit instruction for every active lane. `codes` are the source codes (one per operand).
#[allow(clippy::too_many_arguments)]
pub fn execute(
    wave: &mut Wave,
    name: &str,
    codes: &[u32],
    vdst: u32,
    literal: Option<u32>,
    abs_bits: u32,
    neg_bits: u32,
    clamp: bool,
    omod: u32,
) -> Result<bool, Error> {
    let Some((kinds, result)) = spec(name) else {
        return Ok(false);
    };
    for lane in wave.active()? {
        let mut values = Vec::with_capacity(kinds.len());
        for (index, (kind, code)) in kinds.iter().zip(codes).enumerate() {
            let bits = if kind.is_wide() {
                if *code == 255 {
                    return Err(Error::Untranslated(
                        "littéral sur une source 64 bits".into(),
                    ));
                }
                read64(wave, *code, lane, *kind == Kind::F64)?
            } else {
                u64::from(wave.read_lane(*code, lane, literal)?)
            };
            values.push(modify(bits, *kind, index as u32, abs_bits, neg_bits));
        }
        let Some(value) = compute(name, &values, clamp, omod) else {
            return Err(Error::Untranslated(format!(
                "instruction non traduite : {name} (64 bits)"
            )));
        };
        let width = if result.is_wide() { 2 } else { 1 };
        let destination = wave.vgpr_group(vdst as usize, width)?;
        wave.v[destination][lane] = (value & MASK32) as u32;
        if result.is_wide() {
            wave.v[destination + 1][lane] = ((value >> 32) & MASK32) as u32;
        }
    }

    Ok(true)
}

#[cfg(test)]
mod canonical_nan {
    use super::*;

    #[test]
    fn sqrt_of_minus_one_is_positive_canonical_nan() {
        let bits = compute("v_sqrt_f64", &[(-1.0f64).to_bits()], false, 0);
        assert_eq!(bits, Some(0x7FF8_0000_0000_0000));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rounding_functions_keep_the_sign_of_zero_and_quiet_nans() {
        assert_eq!(rounded(-0.5, f64::trunc).to_bits(), (-0.0f64).to_bits());
        assert!(
            rounded(f64::from_bits(0x7FF0_0000_0000_0001), f64::floor).to_bits() & QUIET_F64 != 0
        );
    }

    #[test]
    fn a_64_bit_result_at_v255_is_refused_without_mutation() {
        // v_lshlrev_b64 v[255:256], 4, v[2:3]: v256 does not exist.
        let mut wave = Wave::new(32).unwrap();
        wave.v[2] = vec![1; 32];
        wave.v[255] = vec![0x1234_5678; 32];
        let codes = [132, 256 + 2];
        let refused = execute(
            &mut wave,
            "v_lshlrev_b64",
            &codes,
            255,
            None,
            0,
            0,
            false,
            0,
        );
        assert!(matches!(refused, Err(Error::Operand(_))));
        assert!(wave.v[255].iter().all(|&value| value == 0x1234_5678));
        // One register lower, the pair v254:v255 fits.
        assert!(
            execute(
                &mut wave,
                "v_lshlrev_b64",
                &codes,
                254,
                None,
                0,
                0,
                false,
                0
            )
            .unwrap()
        );
        assert_eq!((wave.v[254][0], wave.v[255][0]), (0x10, 0));
    }

    #[test]
    fn min_and_max_ignore_a_single_nan() {
        assert_eq!(pick(f64::NAN, 2.0, |x, y| x < y), 2.0);
        assert_eq!(pick(1.0, f64::NAN, |x, y| x < y), 1.0);
        assert_eq!(pick(1.0, 2.0, |x, y| x > y), 2.0);
    }

    #[test]
    fn the_64_bit_shifts_mask_their_amount_to_six_bits() {
        assert_eq!(compute("v_lshlrev_b64", &[64 + 1, 3], false, 0), Some(6));
        assert_eq!(
            compute("v_ashrrev_i64", &[1, u64::MAX], false, 0),
            Some(u64::MAX)
        );
    }
}
