//! Single-precision values as the model holds them: bits, with arithmetic in double precision and
//! the result rounded once to single precision (`emu/rdna2_exec_ops.py`: `float_bits`, `_fma`,
//! `_float_op`, and the Python builtins the table uses: `min`, `max`, `math.frexp`).

/// The bits of Python's `float('nan')` as `float_bits` returns them: a positive quiet NaN.
pub const CANONICAL_NAN: u32 = 0x7FC0_0000;
/// The bits of Python's `math.nan` as a double: a positive quiet NaN. `f64::NAN` is not used for the
/// model's NaN results, because its sign bit is not fixed by the language (x86 gives the negative one).
pub const CANONICAL_NAN64: u64 = 0x7FF8_0000_0000_0000;

/// `math.nan` as a double (see `CANONICAL_NAN64`).
pub fn canonical_nan64() -> f64 {
    f64::from_bits(CANONICAL_NAN64)
}
/// The largest single-precision value below 1.0 (`ALMOST_ONE_F32`).
pub const ALMOST_ONE: u32 = 0x3F7F_FFFF;

/// IEEE 754 single-precision bits of a double. Values beyond the format are ±infinity.
pub fn float_bits(value: f64) -> u32 {
    (value as f32).to_bits()
}

/// The single-precision value of some bits, as a double (exact).
pub fn bits_float(bits: u32) -> f64 {
    f64::from(f32::from_bits(bits))
}

/// Bits of a result of `_float_op`: `None` is the Python `ValueError`, which gives the canonical NaN.
pub fn result_bits(value: Option<f64>) -> u32 {
    value.map_or(CANONICAL_NAN, float_bits)
}

/// The inline float constants of source operands 240 to 248 (the Python `INLINE_FLOATS`).
pub fn inline_constant(code: u32) -> Option<u64> {
    let value = match code {
        240 => 0.5,
        241 => -0.5,
        242 => 1.0,
        243 => -1.0,
        244 => 2.0,
        245 => -2.0,
        246 => 4.0,
        247 => -4.0,
        248 => 0.159_154_943_091_895_35,
        _ => return None,
    };
    Some(u64::from(float_bits(value)))
}

/// The double-precision bit patterns of inline float operands 240 to 248.
pub fn inline_double(code: u32) -> Option<u64> {
    Some(match code {
        240 => 0x3fe0_0000_0000_0000,
        241 => 0xbfe0_0000_0000_0000,
        242 => 0x3ff0_0000_0000_0000,
        243 => 0xbff0_0000_0000_0000,
        244 => 0x4000_0000_0000_0000,
        245 => 0xc000_0000_0000_0000,
        246 => 0x4010_0000_0000_0000,
        247 => 0xc010_0000_0000_0000,
        248 => 0x3fc4_5f30_6dc9_c882,
        _ => return None,
    })
}

/// Python's `min(a, b)`: `a` unless `b` is smaller (so a NaN in `a` wins).
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python's `max(a, b)`: `a` unless `b` is larger.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `a × b + c` of the model's `_mul_add` (`v_mad_f32`, `v_mac_f32`, `v_madmk_f32`, `v_madak_f32`): the
/// product, then the sum, in double precision, rounded to single precision afterwards by `float_bits`.
///
/// The NaN it returns is the one the model's x86-64 arithmetic gives for `a * b + c`: the first NaN
/// operand of the product (`a`, then `b`), or the NaN the product makes (∞ × 0); only then a NaN `c`.
/// That order is written out because a plain `a * b + c` leaves it to the compiler, which may swap the
/// operands of the addition: it did, and the addend's NaN came out where the model keeps the product's.
/// Without a NaN operand the expression is computed as it is (∞ − ∞ gives the processor's NaN, as in
/// the model).
pub fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    let product = if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a * b
    };
    if product.is_nan() {
        return product;
    }
    if c.is_nan() {
        return c;
    }
    product + c
}

/// `math.frexp`: `x = m * 2^e` with 0.5 <= |m| < 1 (zero, infinities and NaN return `(x, 0)`).
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    if exponent == 0 {
        // Subnormal: scale into the normal range first.
        let (mantissa, scaled) = frexp(x * 2f64.powi(64));
        return (mantissa, scaled - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7FF << 52)) | (1022 << 52));
    (mantissa, exponent - 1022)
}

/// `a × b + c` rounded once to single precision (`_fma` of the model): the sum is rounded to odd
/// when it is inexact, then the single-precision rounding of `float_bits` is the exact one.
/// NaN operands are returned as they are; an infinite or NaN sum without a NaN operand is NaN.
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    let product = a * b;
    let total = product + c;
    if total.is_nan() || total.is_infinite() {
        return first_nan(a, b, c, total);
    }
    let back = total - product;
    let error = (product - (total - back)) + (c - back);
    let (mantissa, _) = frexp(total);
    let parity = (mantissa * 2f64.powi(53)).rem_euclid(2.0);
    if error != 0.0 && parity == 0.0 {
        return if error > 0.0 {
            total.next_up()
        } else {
            total.next_down()
        };
    }
    total
}

/// The NaN of `a × b + c` when the sum is not finite: the first NaN operand, else the sum if it
/// is infinite, else the canonical NaN.
fn first_nan(a: f64, b: f64, c: f64, total: f64) -> f64 {
    for operand in [a, b, c] {
        if operand.is_nan() {
            return operand;
        }
    }
    if total.is_nan() {
        canonical_nan64()
    } else {
        total
    }
}

/// `legacy` fused multiply-add: `z` when either product factor is zero (DX9 rules), else `fma`.
pub fn legacy_fma(x: f64, y: f64, z: f64) -> f64 {
    if x == 0.0 || y == 0.0 {
        z
    } else {
        fma(x, y, z)
    }
}

/// `ldexp` of the model: x × 2^n with the model's overflow (±infinity) and underflow (±0) results.
pub fn ldexp(x: f64, n: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    if n > 2200 {
        return f64::INFINITY.copysign(x);
    }
    if n < -2200 {
        return 0f64.copysign(x);
    }
    let mut value = x;
    let mut remaining = n;
    while remaining > 1000 {
        value *= 2f64.powi(1000);
        remaining -= 1000;
    }
    while remaining < -1000 {
        value *= 2f64.powi(-1000);
        remaining += 1000;
    }
    value * 2f64.powi(remaining as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_inline_constants_are_their_single_precision_bits() {
        assert_eq!(inline_constant(242), Some(0x3F80_0000));
        assert_eq!(inline_constant(241), Some(0xBF00_0000));
        assert_eq!(inline_constant(248), Some(0x3E22_F983));
        assert_eq!(inline_constant(249), None);
    }

    #[test]
    fn values_beyond_the_format_are_infinities() {
        assert_eq!(float_bits(1e300), 0x7F80_0000);
        assert_eq!(float_bits(-1e300), 0xFF80_0000);
        assert_eq!(bits_float(0x3F80_0000), 1.0);
    }

    #[test]
    fn frexp_splits_normal_subnormal_and_special_values() {
        assert_eq!(frexp(8.0), (0.5, 4));
        assert_eq!(frexp(-0.75), (-0.75, 0));
        assert_eq!(frexp(f64::from_bits(1)), (0.5, -1073));
        assert_eq!(frexp(0.0), (0.0, 0));
    }

    #[test]
    fn min_and_max_keep_the_first_operand_on_nan_like_python() {
        assert!(py_min(f64::NAN, 1.0).is_nan());
        assert_eq!(py_min(1.0, f64::NAN), 1.0);
        assert_eq!(py_max(2.0, 3.0), 3.0);
    }

    #[test]
    fn mul_add_keeps_the_nan_of_the_product_before_the_addend() {
        let (a, b, c) = (
            f64::from_bits(0x7FF8_0000_0000_0A0A),
            f64::from_bits(0x7FF8_0000_0000_0B0B),
            f64::from_bits(0x7FF8_0000_0000_0C0C),
        );
        assert_eq!(mul_add(a, b, c).to_bits(), a.to_bits());
        assert_eq!(mul_add(1.0, b, c).to_bits(), b.to_bits());
        assert_eq!(mul_add(1.0, 2.0, c).to_bits(), c.to_bits());
        // The NaN of ∞ × 0 comes before a NaN addend, as in `(a * b) + c`.
        let generated = mul_add(f64::INFINITY, 0.0, c);
        assert!(generated.is_nan() && generated.to_bits() != c.to_bits());
        assert_eq!(mul_add(2.0, 3.0, 1.0), 7.0);
    }

    #[test]
    fn fma_rounds_the_exact_sum_once() {
        // 1 + 2^-24 is a tie for single precision: the exact sum rounds to even (1.0).
        assert_eq!(float_bits(fma(1.0, 1.0, 2f64.powi(-24))), 0x3F80_0000);
        assert_eq!(
            float_bits(fma(f64::INFINITY, 1.0, -f64::INFINITY)),
            CANONICAL_NAN
        );
    }
}
