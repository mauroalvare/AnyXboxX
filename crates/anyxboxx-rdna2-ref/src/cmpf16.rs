//! Half-precision (f16) vector compares of VOPC and VOP3 (`v_cmp[x]_*_f16`, `v_cmp[x]_class_f16`),
//! ported from `emu/rdna2_exec_cmpf16.py`.

use crate::compare::{FLOAT_NAMES, Test, float_class, float_test, test_of};
use crate::error::Error;
use crate::half::{HALF_SIGN, half_to_f64, read_half};
use crate::vop3::Fields;
use crate::wave::{EXEC_LO, VCC_LO, VGPR_FIRST, Wave};

/// Source codes of the DPP8, DPP8_FI, SDWA and DPP16 forms: refused for f16 compares.
const EXTENSION_CODES: [u32; 4] = [0xE9, 0xEA, 0xF9, 0xFA];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Predicate(Test),
    Class,
}

fn split(name: &str) -> Option<(bool, Kind)> {
    let (writes_exec, rest) = match name.strip_prefix("v_cmpx_") {
        Some(rest) => (true, rest),
        None => (false, name.strip_prefix("v_cmp_")?),
    };
    let test = rest.strip_suffix("_f16")?;
    if test == "class" {
        return Some((writes_exec, Kind::Class));
    }
    if FLOAT_NAMES.contains(&test) {
        return Some((writes_exec, Kind::Predicate(test_of(test)?)));
    }
    None
}

/// Whether `name` is one of the 34 half-precision compare instructions (`COMPAREF16`).
pub fn supports(name: &str) -> bool {
    split(name).is_some()
}

fn evaluate(kind: Kind, a: u32, b: u32) -> bool {
    match kind {
        Kind::Predicate(test) => float_test(test, half_to_f64(a), half_to_f64(b)),
        Kind::Class => (b >> float_class(u64::from(a), 16)) & 1 == 1,
    }
}

fn modify_half(mut bits: u32, abs: u32, neg: u32, index: u32) -> u32 {
    if (abs >> index) & 1 != 0 {
        bits &= !HALF_SIGN;
    }
    if (neg >> index) & 1 != 0 {
        bits ^= HALF_SIGN;
    }
    bits
}

/// Executes a VOPC half-precision compare: writes `VCC` (`v_cmp`) or `EXEC` (`v_cmpx`).
pub fn execute(
    wave: &mut Wave,
    name: &str,
    src0: u32,
    vsrc1: u32,
    literal: Option<u32>,
) -> Result<(), Error> {
    if EXTENSION_CODES.contains(&src0) {
        return Err(Error::Untranslated(format!(
            "{name}: SDWA/DPP form (src0 = 0x{src0:X}) not supported for f16"
        )));
    }
    let (writes_exec, kind) = split(name).ok_or_else(|| Error::Untranslated(name.to_string()))?;
    let src1 = VGPR_FIRST + vsrc1;
    let mut mask = 0u64;
    for lane in wave.active()? {
        let a = read_half(wave, src0, lane, literal, false)?;
        let b = read_half(wave, src1, lane, literal, false)?;
        mask |= u64::from(evaluate(kind, a, b)) << lane;
    }
    wave.write_mask(if writes_exec { EXEC_LO } else { VCC_LO }, mask)
}

fn check_vop3(name: &str, kind: Kind, fields: &Fields, op_sel: u32) -> Result<(), Error> {
    if op_sel != 0 {
        return Err(Error::Untranslated(format!(
            "{name} with op_sel (VOP3 compare, 16-bit sources): not usable per the RDNA 2 ISA"
        )));
    }
    if fields.clamp || fields.omod != 0 {
        return Err(Error::Untranslated(format!(
            "{name} with clamp or omod (VOP3 f16 compare): not measured on RDNA 2"
        )));
    }
    if kind == Kind::Class && ((fields.abs | fields.neg) >> 1) & 1 != 0 {
        return Err(Error::Untranslated(format!(
            "{name} with abs/neg on the class mask S1: not measured on RDNA 2"
        )));
    }
    Ok(())
}

/// Executes a VOP3 half-precision compare: writes `fields.vdst` (`v_cmp`) or `EXEC` (`v_cmpx`).
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    word: u64,
    literal: Option<u32>,
) -> Result<(), Error> {
    let (writes_exec, kind) = split(name).ok_or_else(|| Error::Untranslated(name.to_string()))?;
    let fields = Fields::new(word);
    let op_sel = ((word >> 11) & 0xF) as u32;
    check_vop3(name, kind, &fields, op_sel)?;
    let mut mask = 0u64;
    for lane in wave.active()? {
        let raw_a = read_half(wave, fields.sources[0], lane, literal, false)?;
        let raw_b = read_half(wave, fields.sources[1], lane, literal, false)?;
        let a = modify_half(raw_a, fields.abs, fields.neg, 0);
        let b = if kind == Kind::Class {
            raw_b
        } else {
            modify_half(raw_b, fields.abs, fields.neg, 1)
        };
        mask |= u64::from(evaluate(kind, a, b)) << lane;
    }
    wave.write_mask(if writes_exec { EXEC_LO } else { fields.vdst }, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_all_thirty_four_half_compares() {
        assert!(supports("v_cmp_f_f16"));
        assert!(supports("v_cmp_tru_f16"));
        assert!(supports("v_cmpx_neq_f16"));
        assert!(supports("v_cmp_class_f16"));
        assert!(supports("v_cmpx_class_f16"));
        assert!(!supports("v_cmp_ne_f16"));
        assert!(!supports("v_cmp_eq_f32"));
    }

    #[test]
    fn predicates_and_class_mask_match_ieee_binary16() {
        let (_, eq) = split("v_cmp_eq_f16").unwrap();
        assert!(evaluate(eq, 0x0000, 0x8000));
        assert!(!evaluate(eq, 0x7E00, 0x7E00));
        let (_, neq) = split("v_cmp_neq_f16").unwrap();
        assert!(evaluate(neq, 0x7E00, 0x3C00));
        let (_, cls) = split("v_cmp_class_f16").unwrap();
        assert!(evaluate(cls, 0x7C01, 1 << 0)); // signaling NaN
        assert!(evaluate(cls, 0x7E00, 1 << 1)); // quiet NaN
        assert!(evaluate(cls, 0xFC00, 1 << 2)); // -infinity
        assert!(evaluate(cls, 0x0001, 1 << 7)); // +subnormal
        assert!(evaluate(cls, 0x7C00, 1 << 9)); // +infinity
    }
}
