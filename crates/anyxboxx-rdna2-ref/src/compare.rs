//! VOPC instructions (vector compares): port of `VECTOR_COMPARE` and `COMPARE64` of the Python model
//! (`emu/rdna2_exec_ops.py`, `emu/rdna2_exec_cmp64.py`).
//!
//! `v_cmp_<test>_f32/u32/i32` compare one lane each; `v_cmp_<test>_f64/u64/i64` read 64-bit operands
//! (register pairs or inline constants), and `v_cmp_class_f32/f64` test the IEEE class of the first
//! source against the bit mask of the second. The `v_cmpx` forms write EXEC, the others VCC.

use crate::decode::LITERAL;
use crate::error::Error;
use crate::float::bits_float;
use crate::scalar::sign_extend;
use crate::wave::{EXEC_LO, VCC_LO, VGPR_FIRST, Wave};

const MASK64: u64 = u64::MAX;

/// The comparison of two values, in the model's order: `lt`, `eq`, `le`, `gt`, `lg`, `ge`, and the
/// unordered and negated forms. Integer compares only use `f`, `lt`, `eq`, `le`, `gt`, `ne`, `ge`, `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Test {
    False,
    True,
    Lt,
    Eq,
    Le,
    Gt,
    Lg,
    Ge,
    Ne,
    Ordered,
    Unordered,
    NotGe,
    NotLg,
    NotGt,
    NotLe,
    NotEq,
    NotLt,
}

pub(crate) fn test_of(name: &str) -> Option<Test> {
    Some(match name {
        "f" => Test::False,
        "tru" | "t" => Test::True,
        "lt" => Test::Lt,
        "eq" => Test::Eq,
        "le" => Test::Le,
        "gt" => Test::Gt,
        "lg" => Test::Lg,
        "ge" => Test::Ge,
        "ne" => Test::Ne,
        "o" => Test::Ordered,
        "u" => Test::Unordered,
        "nge" => Test::NotGe,
        "nlg" => Test::NotLg,
        "ngt" => Test::NotGt,
        "nle" => Test::NotLe,
        "neq" => Test::NotEq,
        "nlt" => Test::NotLt,
        _ => return None,
    })
}

/// The float tests of `FLOAT_TESTS` (the Python on doubles; NaN makes every ordered test false).
#[allow(clippy::neg_cmp_op_on_partial_ord, clippy::nonminimal_bool)] // negated forms are the IEEE unordered tests on purpose
pub(crate) fn float_test(test: Test, x: f64, y: f64) -> bool {
    match test {
        Test::False => false,
        Test::True => true,
        Test::Lt => x < y,
        Test::Eq => x == y,
        Test::Le => x <= y,
        Test::Gt => x > y,
        Test::Lg => x != y && !x.is_nan() && !y.is_nan(),
        Test::Ge => x >= y,
        Test::Ordered => !(x.is_nan() || y.is_nan()),
        Test::Unordered => x.is_nan() || y.is_nan(),
        Test::NotGe => !(x >= y),
        Test::NotLg => x.is_nan() || y.is_nan() || x == y,
        Test::NotGt => !(x > y),
        Test::NotLe => !(x <= y),
        Test::NotEq => !(x == y),
        Test::NotLt => !(x < y),
        Test::Ne => x != y,
    }
}

/// The integer tests of `INTEGER_TESTS`.
fn integer_test(test: Test, x: i128, y: i128) -> bool {
    match test {
        Test::False => false,
        Test::True => true,
        Test::Lt => x < y,
        Test::Eq => x == y,
        Test::Le => x <= y,
        Test::Gt => x > y,
        Test::Ne => x != y,
        Test::Ge => x >= y,
        _ => false,
    }
}

/// The float tests (`FLOAT_TESTS`) and the integer tests (`INTEGER_TESTS`) by name.
pub(crate) const FLOAT_NAMES: [&str; 16] = [
    "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt",
    "tru",
];
const INTEGER_NAMES: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];

/// The name of a compare, split into (writes EXEC, test, operand type). `None` for any other name.
fn split(name: &str) -> Option<(bool, Test, &'static str)> {
    let (writes_exec, rest) = match name.strip_prefix("v_cmpx_") {
        Some(rest) => (true, rest),
        None => (false, name.strip_prefix("v_cmp_")?),
    };
    let (test, kind) = rest.rsplit_once('_')?;
    match (test, kind) {
        ("class", "f32") => return Some((writes_exec, Test::True, "class32")),
        ("class", "f64") => return Some((writes_exec, Test::True, "class64")),
        _ => {}
    }
    let kind = match kind {
        "f32" if FLOAT_NAMES.contains(&test) => "f32",
        "f64" if FLOAT_NAMES.contains(&test) => "f64",
        "u32" if INTEGER_NAMES.contains(&test) => "u32",
        "i32" if INTEGER_NAMES.contains(&test) => "i32",
        "u16" if INTEGER_NAMES[1..7].contains(&test) => "u16",
        "i16" if INTEGER_NAMES[1..7].contains(&test) => "i16",
        "u64" if INTEGER_NAMES.contains(&test) => "u64",
        "i64" if INTEGER_NAMES.contains(&test) => "i64",
        _ => return None,
    };
    Some((writes_exec, test_of(test)?, kind))
}

/// Whether the name is a VOPC compare of this model (32-bit or 64-bit, class included).
pub fn is_compare(name: &str) -> bool {
    split(name).is_some()
}

/// IEEE class bit index of a float (0 to 9), as `float_class` of the model.
pub(crate) fn float_class(bits: u64, width: u32) -> u32 {
    let (exponent_bits, mantissa_bits) = match width {
        16 => (5, 10),
        32 => (8, 23),
        _ => (11, 52),
    };
    let negative = (bits >> (width - 1)) & 1 == 1;
    let exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1);
    let mantissa = bits & ((1 << mantissa_bits) - 1);
    let all_ones = (1 << exponent_bits) - 1;
    if exponent == all_ones {
        if mantissa != 0 {
            return if (mantissa >> (mantissa_bits - 1)) & 1 == 1 {
                1
            } else {
                0
            };
        }
        return if negative { 2 } else { 9 };
    }
    if exponent == 0 {
        if mantissa == 0 {
            return if negative { 5 } else { 6 };
        }
        return if negative { 4 } else { 7 };
    }
    if negative { 3 } else { 8 }
}

/// A 64-bit source of one lane: a register pair (code 256 and up), an SGPR pair, or an inline constant.
pub fn read64(wave: &Wave, code: u32, lane: usize, is_float: bool) -> Result<u64, Error> {
    if code >= VGPR_FIRST {
        let register = (code - VGPR_FIRST) as usize;
        if register + 1 >= wave.v.len() {
            return Err(Error::Operand(format!(
                "paire de registres vectoriels hors plage : v{register}"
            )));
        }
        return Ok(u64::from(wave.v[register][lane]) | u64::from(wave.v[register + 1][lane]) << 32);
    }
    if code < 128 || (128..=208).contains(&code) {
        return Ok(wave.read_scalar(code, None, 2)? & MASK64);
    }
    match crate::float::inline_double(code) {
        Some(value) if is_float => Ok(value),
        _ => Err(Error::Untranslated(format!(
            "opérande 64 bits {code} non pris en charge"
        ))),
    }
}

/// Executes a VOPC compare of the model. `Ok(false)` when `name` is not one of them.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    src0: u32,
    vsrc1: u32,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if !is_compare(name) {
        return Ok(false);
    }
    let (writes_exec, test, kind) = split(name).ok_or_else(|| Error::Operand(name.to_string()))?;
    let destination = if writes_exec { EXEC_LO } else { VCC_LO };
    let mut mask = 0u64;
    match kind {
        "f32" | "u32" | "i32" | "u16" | "i16" => {
            for lane in wave.active()? {
                let a = wave.read_lane(src0, lane, literal)?;
                let b = wave.v[vsrc1 as usize][lane];
                let hit = match kind {
                    "f32" => float_test(test, bits_float(a), bits_float(b)),
                    "u32" => integer_test(test, i128::from(a), i128::from(b)),
                    "u16" => integer_test(test, i128::from(a as u16), i128::from(b as u16)),
                    "i16" => integer_test(test, i128::from(a as i16), i128::from(b as i16)),
                    _ => integer_test(test, i128::from(a as i32), i128::from(b as i32)),
                };
                mask |= u64::from(hit) << lane;
            }
        }
        "class32" => {
            for lane in wave.active()? {
                let a = wave.read_lane(src0, lane, literal)?;
                let b = wave.v[vsrc1 as usize][lane];
                mask |= u64::from((b >> float_class(u64::from(a), 32)) & 1) << lane;
            }
        }
        "class64" => {
            if src0 == LITERAL {
                return Err(Error::Untranslated(format!(
                    "littéral dans une comparaison 64 bits : {name}"
                )));
            }
            for lane in wave.active()? {
                let a = read64(wave, src0, lane, true)?;
                let b = u64::from(wave.v[vsrc1 as usize][lane]);
                mask |= ((b >> float_class(a, 64)) & 1) << lane;
            }
        }
        _ => {
            if src0 == LITERAL {
                return Err(Error::Untranslated(format!(
                    "littéral dans une comparaison 64 bits : {name}"
                )));
            }
            let vector = VGPR_FIRST + vsrc1;
            for lane in wave.active()? {
                let a = read64(wave, src0, lane, kind == "f64")?;
                let b = read64(wave, vector, lane, kind == "f64")?;
                let hit = match kind {
                    "f64" => float_test(test, f64::from_bits(a), f64::from_bits(b)),
                    "u64" => integer_test(test, i128::from(a), i128::from(b)),
                    _ => integer_test(
                        test,
                        i128::from(sign_extend(a, 64)),
                        i128::from(sign_extend(b, 64)),
                    ),
                };
                mask |= u64::from(hit) << lane;
            }
        }
    }
    wave.write_mask(destination, mask)?;
    Ok(true)
}

/// A compare on raw bits.
pub type Compare32 = Box<dyn Fn(u32, u32) -> bool>;

/// The 32-bit compare `name` (`v_cmp`/`v_cmpx` of f32, u32 or i32, the `VECTOR_COMPARE` forms) as a test
/// on raw bits, and whether it writes EXEC. `None` for the other compares.
pub fn compare32(name: &str) -> Option<(bool, Compare32)> {
    let (writes_exec, test, kind) = split(name)?;
    let compare: Box<dyn Fn(u32, u32) -> bool> = match kind {
        "f32" => Box::new(move |a, b| float_test(test, bits_float(a), bits_float(b))),
        "u32" => Box::new(move |a, b| integer_test(test, i128::from(a), i128::from(b))),
        "u16" => {
            Box::new(move |a, b| integer_test(test, i128::from(a as u16), i128::from(b as u16)))
        }
        "i16" => {
            Box::new(move |a, b| integer_test(test, i128::from(a as i16), i128::from(b as i16)))
        }
        "i32" => {
            Box::new(move |a, b| integer_test(test, i128::from(a as i32), i128::from(b as i32)))
        }
        _ => return None,
    };
    Some((writes_exec, compare))
}

/// Input modifier of one source of a 64-bit (or class) compare, as the model's `_modify`.
fn modify64(kind: &str, index: u32, value: u64, abs_bits: u32, neg_bits: u32) -> u64 {
    if matches!(kind, "u64" | "i64") || (kind.starts_with("class") && index == 1) {
        return value;
    }
    let sign = if kind == "class32" {
        1u64 << 31
    } else {
        1u64 << 63
    };
    let mut out = value;
    if abs_bits >> index & 1 == 1 {
        out &= !sign;
    }
    if neg_bits >> index & 1 == 1 {
        out ^= sign;
    }
    out
}

/// The VOP3 form of a 64-bit or class compare: any source codes, the input modifiers of the model, and the
/// destination SGPR named by vdst (EXEC for the v_cmpx forms). `Ok(false)` for the other compares.
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    sources: [u32; 2],
    literal: Option<u32>,
    destination: u32,
    abs_bits: u32,
    neg_bits: u32,
) -> Result<bool, Error> {
    let Some((writes_exec, test, kind)) = split(name) else {
        return Ok(false);
    };
    if !matches!(kind, "f64" | "u64" | "i64" | "class32" | "class64") {
        return Ok(false);
    }
    let wide: &[u32] = match kind {
        "class32" => &[],
        "class64" => &sources[..1],
        _ => &sources,
    };
    if wide.contains(&LITERAL) {
        return Err(Error::Untranslated(format!(
            "littéral dans une comparaison 64 bits : {name}"
        )));
    }
    let mut mask = 0u64;
    for lane in wave.active()? {
        let (a, b) = match kind {
            "class32" => (
                u64::from(wave.read_lane(sources[0], lane, literal)?),
                u64::from(wave.read_lane(sources[1], lane, literal)?),
            ),
            "class64" => (
                read64(wave, sources[0], lane, true)?,
                u64::from(wave.read_lane(sources[1], lane, literal)?),
            ),
            _ => {
                let is_float = kind == "f64";
                (
                    read64(wave, sources[0], lane, is_float)?,
                    read64(wave, sources[1], lane, is_float)?,
                )
            }
        };
        let a = modify64(kind, 0, a, abs_bits, neg_bits);
        let b = modify64(kind, 1, b, abs_bits, neg_bits);
        let hit = match kind {
            "class32" => (b >> float_class(a, 32)) & 1 == 1,
            "class64" => (b >> float_class(a, 64)) & 1 == 1,
            "f64" => float_test(test, f64::from_bits(a), f64::from_bits(b)),
            "u64" => integer_test(test, i128::from(a), i128::from(b)),
            _ => integer_test(
                test,
                i128::from(sign_extend(a, 64)),
                i128::from(sign_extend(b, 64)),
            ),
        };
        mask |= u64::from(hit) << lane;
    }
    let target = if writes_exec { EXEC_LO } else { destination };
    wave.write_mask(target, mask)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_split_into_exec_test_and_type() {
        assert_eq!(split("v_cmp_lt_f32"), Some((false, Test::Lt, "f32")));
        assert_eq!(split("v_cmpx_ne_i32"), Some((true, Test::Ne, "i32")));
        assert!(!is_compare("v_cmp_ne_f32"));
        assert!(is_compare("v_cmp_class_f64"));
        assert!(!is_compare("v_add_f32"));
    }

    #[test]
    fn nan_makes_ordered_tests_false_and_unordered_true() {
        assert!(!float_test(Test::Lt, f64::NAN, 1.0));
        assert!(float_test(Test::NotLt, f64::NAN, 1.0));
        assert!(float_test(Test::Unordered, f64::NAN, 1.0));
    }

    #[test]
    fn the_class_bit_index_follows_the_ieee_classes() {
        assert_eq!(float_class(0x7F80_0000, 32), 9); // +infinity
        assert_eq!(float_class(0xFF80_0000, 32), 2); // -infinity
        assert_eq!(float_class(0x7FC0_0000, 32), 1); // quiet NaN
        assert_eq!(float_class(0x0000_0001, 32), 7); // +subnormal
        assert_eq!(float_class(0x8000_0000, 32), 5); // -0
        assert_eq!(float_class(1.0f64.to_bits(), 64), 8); // +normal
    }
}
