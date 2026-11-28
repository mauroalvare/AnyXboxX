//! Vector instructions of the VOP1 and VOP2 encodings: port of the `VECTOR` table of
//! `emu/rdna2_exec_ops.py` (the operations that take a source, a second source, the old destination,
//! the literal and the VCC bit), plus the VOP2 carry chain of `rdna2_exec_vop3.CARRY`.
//!
//! Each lane computes `operation(a, b, d, k, c)` and the result is written masked to 32 bits. The
//! float operations follow the Python model's exception paths (`ValueError` gives the canonical NaN),
//! and its `min`/`max`/`%` conventions, not the IEEE ones.

use crate::error::Error;
use crate::float::{self, ALMOST_ONE, bits_float, float_bits, py_max, py_min, result_bits};
use crate::scalar::sign_extend;
use crate::wave::{VCC_LO, Wave};

const MASK32: u64 = 0xFFFF_FFFF;
const INT32_MIN: f64 = -2147483648.0;
const INT32_MAX: f64 = 2147483647.0;
const UINT32_MAX: f64 = 4294967295.0;

/// The operations that write no result (the Python `VECTOR_NO_OPERATION`).
pub fn is_no_operation(name: &str) -> bool {
    matches!(name, "v_nop" | "v_pipeflush" | "v_clrexcp")
}

/// Whether `name` is a VOP2 carry-chain operation (the Python `VOP2_CARRY`).
pub fn is_carry(name: &str) -> bool {
    matches!(
        name,
        "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
    )
}

/// Carry-chain operations of VOP2: result and carry out.
pub fn carry(name: &str, a: u64, b: u64, carry_in: u64) -> Option<(u64, bool)> {
    Some(match name {
        "v_add_co_ci_u32" => (a + b + carry_in, (a + b + carry_in) >> 32 != 0),
        "v_sub_co_ci_u32" => (a.wrapping_sub(b).wrapping_sub(carry_in), a < b + carry_in),
        "v_subrev_co_ci_u32" => (b.wrapping_sub(a).wrapping_sub(carry_in), b < a + carry_in),
        "v_add_co_u32" => (a + b, (a + b) >> 32 != 0),
        "v_sub_co_u32" => (a.wrapping_sub(b), a < b),
        "v_subrev_co_u32" => (b.wrapping_sub(a), b < a),
        _ => return None,
    })
}

/// `_to_int` of the model: NaN gives 0, infinities the bound they point to, else the truncated value
/// clamped to the range.
fn to_int(value: f64, low: f64, high: f64) -> u32 {
    let clamped = if value.is_nan() {
        return 0;
    } else if value.is_infinite() {
        if value > 0.0 { high } else { low }
    } else {
        value.trunc().clamp(low, high)
    };
    (clamped as i64) as u32
}

/// `_round_to_int`: `rounding` applied to the exact value, saturated; a NaN gives the sign's bound.
fn round_to_int(bits: u32, rounding: fn(f64) -> f64) -> u32 {
    let x = bits_float(bits);
    if x.is_nan() {
        return if bits >> 31 == 1 {
            INT32_MIN as i64 as u32
        } else {
            INT32_MAX as i64 as u32
        };
    }
    let rounded = if x.is_finite() { rounding(x) } else { x };
    to_int(rounded, INT32_MIN, INT32_MAX)
}

/// `_signed_round`: rounding keeps the sign of the input; ±infinity is unchanged; NaN is a `ValueError`.
fn signed_round(x: f64, rounding: fn(f64) -> f64) -> Option<f64> {
    if x.is_infinite() {
        return Some(x);
    }
    if x.is_nan() {
        return None;
    }
    Some(rounding(x).copysign(x))
}

fn reciprocal(x: f64) -> f64 {
    if x == 0.0 {
        f64::INFINITY.copysign(x)
    } else {
        1.0 / x
    }
}

/// `math.sqrt`: a `ValueError` for a negative input (or -infinity); NaN passes through.
fn sqrt(x: f64) -> Option<f64> {
    if x < 0.0 || x == f64::NEG_INFINITY {
        None
    } else {
        Some(x.sqrt())
    }
}

/// `math.log2` as `_log2`: zero gives -infinity, a negative input is a `ValueError`.
fn log2(x: f64) -> Option<f64> {
    if x == 0.0 {
        Some(f64::NEG_INFINITY)
    } else if x < 0.0 {
        None
    } else {
        Some(x.log2())
    }
}

/// `_fract`: x - floor(x), at most the largest value below 1.0; NaN is a `ValueError`, ±infinity a NaN.
fn fract(x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    if x.is_infinite() {
        return Some(crate::float::canonical_nan64());
    }
    let floor = signed_round(x, f64::floor)?;
    Some(py_min(x - floor, bits_float(ALMOST_ONE)))
}

/// `_exp2`: 2^x, +infinity past the range (Python's `OverflowError`).
fn exp2(x: f64) -> f64 {
    2f64.powf(x)
}

/// `_round_even`: infinities and NaN are returned as they are.
fn round_even(x: f64) -> f64 {
    if x.is_finite() {
        x.round_ties_even().copysign(x)
    } else {
        x
    }
}

/// `(x × 2) × π` as the model computes the sine and cosine arguments; an infinite argument is a `ValueError`.
fn turns(x: f64) -> Option<f64> {
    let angle = x * 2.0 * std::f64::consts::PI;
    if angle.is_infinite() {
        None
    } else {
        Some(angle)
    }
}

/// The 32-bit result of a float operation on bit patterns.
fn unary(a: u32, operation: fn(f64) -> Option<f64>) -> u32 {
    result_bits(operation(bits_float(a)))
}

fn binary(a: u32, b: u32, operation: fn(f64, f64) -> f64) -> u32 {
    float_bits(operation(bits_float(a), bits_float(b)))
}

fn cvt_float(value: f64) -> u32 {
    float_bits(value)
}

/// The bit pattern of a finite non-negative integer value, as the model writes it.
fn exponent(bits: u32) -> u32 {
    let x = bits_float(bits);
    if x.is_nan() || x.is_infinite() {
        0
    } else {
        float::frexp(x).1 as u32
    }
}

fn mantissa(bits: u32) -> u32 {
    let x = bits_float(bits);
    if x == 0.0 || x.is_infinite() || x.is_nan() {
        bits
    } else {
        float_bits(float::frexp(x).0)
    }
}

/// Evaluates one operation for one lane. `None` when `name` is not in the table.
pub fn compute(name: &str, a: u32, b: u32, d: u32, k: u32, c: bool) -> Option<u32> {
    let (a64, b64) = (u64::from(a), u64::from(b));
    let result: u64 = match name {
        "v_mov_b32" | "v_readfirstlane_b32" => a64,
        "v_not_b32" => u64::from(!a),
        "v_bfrev_b32" => u64::from(a.reverse_bits()),
        "v_ffbl_b32" => {
            if a == 0 {
                MASK32
            } else {
                u64::from(a.trailing_zeros())
            }
        }
        "v_ffbh_u32" => {
            if a == 0 {
                MASK32
            } else {
                u64::from(a.leading_zeros())
            }
        }
        "v_ffbh_i32" => {
            let folded = if a >> 31 == 1 { !a } else { a };
            if folded == 0 {
                MASK32
            } else {
                u64::from(folded.leading_zeros())
            }
        }
        "v_cvt_f32_i32" => u64::from(cvt_float(f64::from(a as i32))),
        "v_cvt_f32_u32" => u64::from(cvt_float(f64::from(a))),
        "v_cvt_u32_f32" => u64::from(to_int(bits_float(a), 0.0, UINT32_MAX)),
        "v_cvt_i32_f32" => u64::from(to_int(bits_float(a), INT32_MIN, INT32_MAX)),
        "v_cvt_f32_ubyte0" => u64::from(cvt_float(f64::from(a & 0xFF))),
        "v_cvt_f32_ubyte1" => u64::from(cvt_float(f64::from(a >> 8 & 0xFF))),
        "v_cvt_f32_ubyte2" => u64::from(cvt_float(f64::from(a >> 16 & 0xFF))),
        "v_cvt_f32_ubyte3" => u64::from(cvt_float(f64::from(a >> 24 & 0xFF))),
        "v_rcp_f32" | "v_rcp_iflag_f32" => u64::from(float_bits(reciprocal(bits_float(a)))),
        "v_rsq_f32" => u64::from(result_bits(sqrt(bits_float(a)).map(reciprocal))),
        "v_sqrt_f32" => u64::from(unary(a, sqrt)),
        "v_exp_f32" => u64::from(float_bits(exp2(bits_float(a)))),
        "v_log_f32" => u64::from(unary(a, log2)),
        "v_sin_f32" => u64::from(result_bits(turns(bits_float(a)).map(f64::sin))),
        "v_cos_f32" => u64::from(result_bits(turns(bits_float(a)).map(f64::cos))),
        "v_floor_f32" => u64::from(unary(a, |x| signed_round(x, f64::floor))),
        "v_ceil_f32" => u64::from(unary(a, |x| signed_round(x, f64::ceil))),
        "v_trunc_f32" => u64::from(unary(a, |x| signed_round(x, f64::trunc))),
        "v_rndne_f32" => u64::from(float_bits(round_even(bits_float(a)))),
        "v_fract_f32" => u64::from(unary(a, fract)),
        "v_add_f32" => u64::from(binary(a, b, |x, y| x + y)),
        "v_sub_f32" => u64::from(binary(a, b, |x, y| x - y)),
        "v_subrev_f32" => u64::from(binary(a, b, |x, y| y - x)),
        "v_mul_f32" => u64::from(binary(a, b, |x, y| x * y)),
        "v_mul_legacy_f32" => {
            u64::from(binary(
                a,
                b,
                |x, y| if x == 0.0 || y == 0.0 { 0.0 } else { x * y },
            ))
        }
        "v_min_f32" => u64::from(binary(a, b, py_min)),
        "v_max_f32" => u64::from(binary(a, b, py_max)),
        "v_fmac_f32" => u64::from(float_bits(float::fma(
            bits_float(a),
            bits_float(b),
            bits_float(d),
        ))),
        "v_mac_f32" => u64::from(float_bits(float::mul_add(
            bits_float(a),
            bits_float(b),
            bits_float(d),
        ))),
        "v_fmamk_f32" => u64::from(float_bits(float::fma(
            bits_float(a),
            bits_float(k),
            bits_float(b),
        ))),
        "v_fmaak_f32" => u64::from(float_bits(float::fma(
            bits_float(a),
            bits_float(b),
            bits_float(k),
        ))),
        "v_madmk_f32" => u64::from(float_bits(float::mul_add(
            bits_float(a),
            bits_float(k),
            bits_float(b),
        ))),
        "v_madak_f32" => u64::from(float_bits(float::mul_add(
            bits_float(a),
            bits_float(b),
            bits_float(k),
        ))),
        "v_fmac_legacy_f32" => {
            let (x, y, z) = (bits_float(a), bits_float(b), bits_float(d));
            u64::from(float_bits(float::legacy_fma(x, y, z)))
        }
        "v_cndmask_b32" => {
            if c {
                b64
            } else {
                a64
            }
        }
        "v_add_nc_u32" => a64 + b64,
        "v_sub_nc_u32" => a64.wrapping_sub(b64),
        "v_subrev_nc_u32" => b64.wrapping_sub(a64),
        "v_and_b32" => a64 & b64,
        "v_or_b32" => a64 | b64,
        "v_xor_b32" => a64 ^ b64,
        "v_xnor_b32" => u64::from(!(a ^ b)),
        "v_lshlrev_b32" => b64 << (a & 31),
        "v_lshrrev_b32" => b64 >> (a & 31),
        "v_ashrrev_i32" => (sign_extend(b64, 32) >> (a & 31)) as u64,
        "v_min_u32" => a64.min(b64),
        "v_max_u32" => a64.max(b64),
        "v_min_i32" => sign_extend(a64, 32).min(sign_extend(b64, 32)) as u64,
        "v_max_i32" => sign_extend(a64, 32).max(sign_extend(b64, 32)) as u64,
        "v_mul_u32_u24" => (a64 & 0xFF_FFFF) * (b64 & 0xFF_FFFF),
        "v_mul_hi_u32_u24" => ((a64 & 0xFF_FFFF) * (b64 & 0xFF_FFFF)) >> 32,
        "v_mul_i32_i24" => (sign_extend(a64, 24) * sign_extend(b64, 24)) as u64,
        "v_mul_hi_i32_i24" => ((sign_extend(a64, 24) * sign_extend(b64, 24)) >> 32) as u64,
        "v_cvt_flr_i32_f32" => u64::from(round_to_int(a, f64::floor)),
        "v_cvt_rpi_i32_f32" => u64::from(round_to_int(a, |x| (x + 0.5).floor())),
        "v_cvt_off_f32_i4" => u64::from(cvt_float(sign_extend(a64, 4) as f64 / 16.0)),
        "v_frexp_exp_i32_f32" => u64::from(exponent(a)),
        "v_frexp_mant_f32" => u64::from(mantissa(a)),
        "v_sat_pk_u8_i16" => {
            let saturate = |half: u64| sign_extend(half & 0xFFFF, 16).clamp(0, 0xFF) as u64;
            let low = saturate(a64);
            let high = saturate(a64 >> 16);
            high << 8 | low
        }
        _ => return None,
    };
    Some((result & MASK32) as u32)
}

/// Executes a VOP1 or VOP2 instruction of the table, for every active lane. `Ok(false)` when `name`
/// is not in the table.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if name == "v_readfirstlane_b32" {
        let lane = wave.active()?.first().copied().unwrap_or(0);
        let value = wave.read_lane(fields.src0, lane, literal)?;
        wave.write_scalar(fields.vdst, u64::from(value), 1)?;
        return Ok(true);
    }
    if compute(name, 0, 0, 0, 0, false).is_none() {
        return Ok(false);
    }
    let k = literal.unwrap_or(0);
    let vcc = wave.vcc()?;
    let destination = fields.vdst as usize;
    for lane in wave.active()? {
        let a = wave.read_lane(fields.src0, lane, literal)?;
        let b = match fields.vsrc1 {
            Some(index) => wave.v[index as usize][lane],
            None => 0,
        };
        let old = wave.v[destination][lane];
        let c = vcc >> lane & 1 == 1;
        let value = compute(name, a, b, old, k, c).unwrap_or(0);
        wave.v[destination][lane] = value;
    }
    Ok(true)
}

/// The VOP2 carry chain: the carry of each active lane goes to VCC, and its result to the destination.
pub fn execute_carry(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let vsrc1 = fields.vsrc1.unwrap_or(0) as usize;
    let destination = fields.vdst as usize;
    let carry_in = wave.vcc()?;
    let mut carries = 0u64;
    for lane in wave.active()? {
        let a = u64::from(wave.read_lane(fields.src0, lane, literal)?);
        let b = u64::from(wave.v[vsrc1][lane]);
        let Some((result, carry)) = carry(name, a, b, carry_in >> lane & 1) else {
            return Ok(false);
        };
        wave.v[destination][lane] = (result & MASK32) as u32;
        carries |= u64::from(carry) << lane;
    }
    wave.write_scalar(VCC_LO, carries, wave.mask_width)?;
    Ok(true)
}

/// The fields of a VOP1 or VOP2 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fields {
    pub vdst: u32,
    pub src0: u32,
    pub vsrc1: Option<u32>,
}

/// The fields of a VOP1 (`vsrc1` absent) or VOP2 word.
pub fn fields(family: &str, word: u32) -> Fields {
    let vdst = (word >> 17) & 0xFF;
    let src0 = word & 0x1FF;
    let vsrc1 = (family == "VOP2").then_some((word >> 9) & 0xFF);
    Fields { vdst, src0, vsrc1 }
}

/// Whether `name` is a carry form of VOP3b: the VOP2 carry chain plus the carry-out adds and subtracts.
pub fn is_vop3_carry(name: &str) -> bool {
    is_carry(name) || matches!(name, "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32")
}

/// Whether `name` is a VOP1 or VOP2 operation of the table (its value is computed by `compute`).
pub fn is_known(name: &str) -> bool {
    compute(name, 0, 0, 0, 0, false).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float::CANONICAL_NAN;

    #[test]
    fn integer_operations_mask_and_count_bits_like_the_model() {
        assert_eq!(compute("v_ffbl_b32", 0b1000, 0, 0, 0, false), Some(3));
        assert_eq!(compute("v_ffbh_u32", 1, 0, 0, 0, false), Some(31));
        assert_eq!(compute("v_ffbh_u32", 0, 0, 0, 0, false), Some(0xFFFF_FFFF));
        assert_eq!(
            compute("v_sub_nc_u32", 1, 2, 0, 0, false),
            Some(0xFFFF_FFFF)
        );
        assert_eq!(
            compute("v_ashrrev_i32", 1, 0x8000_0000, 0, 0, false),
            Some(0xC000_0000)
        );
    }

    #[test]
    fn float_conversions_saturate_and_take_nan_like_the_model() {
        assert_eq!(
            compute("v_cvt_u32_f32", float_bits(-3.0), 0, 0, 0, false),
            Some(0)
        );
        assert_eq!(
            compute(
                "v_cvt_i32_f32",
                float_bits(f64::NAN as f32 as f64),
                0,
                0,
                0,
                false
            ),
            Some(0)
        );
        assert_eq!(
            compute("v_cvt_flr_i32_f32", 0x7FC0_0000, 0, 0, 0, false),
            Some(0x7FFF_FFFF)
        );
        assert_eq!(
            compute("v_cvt_flr_i32_f32", 0xFFC0_0000, 0, 0, 0, false),
            Some(0x8000_0000)
        );
    }

    #[test]
    fn sqrt_of_a_negative_is_the_canonical_nan() {
        assert_eq!(
            compute("v_sqrt_f32", float_bits(-4.0), 0, 0, 0, false),
            Some(CANONICAL_NAN)
        );
        assert_eq!(
            compute("v_sqrt_f32", float_bits(4.0), 0, 0, 0, false),
            Some(float_bits(2.0))
        );
    }

    #[test]
    fn cndmask_picks_the_second_source_when_the_vcc_bit_is_set() {
        assert_eq!(compute("v_cndmask_b32", 1, 2, 0, 0, true), Some(2));
        assert_eq!(compute("v_cndmask_b32", 1, 2, 0, 0, false), Some(1));
    }

    #[test]
    fn unknown_names_are_not_computed() {
        assert_eq!(compute("v_no_such_op", 0, 0, 0, 0, false), None);
    }
}
