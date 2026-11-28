//! VOP3 forms of the older encodings: the VOP1, VOP2 and VOPC operations written with VOP3 operands
//! (opcodes below 0x200), the VOP3b carry forms, the lane operations and the 64-bit multiply-add.
//! Port of the matching branches of `execute` in `emu/rdna2_exec_vop3.py`.
//!
//! Input abs and neg apply to the float operations only; clamp and omod apply to float results only.

use crate::error::Error;
use crate::float::{bits_float, float_bits, py_max, py_min};
use crate::scalar::sign_extend;
use crate::vector;
use crate::vop3::Fields;
use crate::wave::{EXEC_LO, NULL, Wave};

const MASK32: u64 = 0xFFFF_FFFF;
const SIGN: u32 = 0x8000_0000;
/// The first opcode of the VOP2 forms, and the first of the VOP1 forms, in the VOP3 opcode space.
const VOP2_START: u32 = 0x100;
const VOP2_END: u32 = 0x140;
const VOP1_START: u32 = 0x180;
const VOP1_END: u32 = 0x200;
/// Results of these names are converted to integers, so they are not float results.
const FLOAT_TO_INT: [&str; 5] = [
    "v_cvt_u32_f32",
    "v_cvt_i32_f32",
    "v_cvt_flr_i32_f32",
    "v_cvt_rpi_i32_f32",
    "v_frexp_exp_i32_f32",
];

/// The Python `float_result`: a float result (clamp and omod apply).
pub fn float_result(name: &str) -> bool {
    name.contains("_f32")
        && !FLOAT_TO_INT.contains(&name)
        && !name.starts_with("v_cvt_u32")
        && !name.starts_with("v_cvt_i32")
}

/// The Python `float_inputs`: float sources (input abs and neg apply).
pub fn float_inputs(name: &str) -> bool {
    float_result(name) || name.starts_with("v_cvt_") || FLOAT_TO_INT.contains(&name)
}

/// Input modifiers of one source (abs, then neg) on a float operand.
pub fn modify(value: u32, index: usize, fields: &Fields, float: bool) -> u32 {
    let mut out = value;
    if float {
        if fields.abs >> index & 1 == 1 {
            out &= !SIGN;
        }
        if fields.neg >> index & 1 == 1 {
            out ^= SIGN;
        }
    }
    out
}

/// Output modifiers of a result: omod (×2, ×4, ÷2) then clamp to [0, 1], on float results only.
pub fn output(fields: &Fields, value: u64, float: bool) -> u32 {
    if !float || !(fields.clamp || fields.omod != 0) {
        return (value & MASK32) as u32;
    }
    let scale = match fields.omod {
        1 => 2.0,
        2 => 4.0,
        3 => 0.5,
        _ => 1.0,
    };
    let mut number = bits_float(value as u32) * scale;
    if fields.clamp && !number.is_nan() {
        number = py_min(py_max(number, 0.0), 1.0);
    }
    float_bits(number)
}

/// The three sources of one lane, with the input modifiers of the float operations.
fn inputs(
    wave: &Wave,
    fields: &Fields,
    lane: usize,
    literal: Option<u32>,
    float: bool,
) -> Result<[u32; 3], Error> {
    let mut values = [0u32; 3];
    for (index, code) in fields.sources.iter().enumerate() {
        values[index] = modify(wave.read_lane(*code, lane, literal)?, index, fields, float);
    }
    Ok(values)
}

/// VOPC forms (opcodes below 0x100): the compare writes EXEC (v_cmpx) or the SGPR named by vdst.
pub fn compare(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let Some((writes_exec, test)) = crate::compare::compare32(name) else {
        return Ok(false);
    };
    let float = name.ends_with("_f32");
    let mut mask = 0u64;
    for lane in wave.active()? {
        let [a, b, _] = inputs(wave, fields, lane, literal, float)?;
        mask |= u64::from(test(a, b)) << lane;
    }
    let destination = if writes_exec { EXEC_LO } else { fields.vdst };
    wave.write_mask(destination, mask)?;
    Ok(true)
}

/// VOP1 and VOP2 forms: the operation of the plain encoding, with VOP3 operands and modifiers.
pub fn operation(
    wave: &mut Wave,
    name: &str,
    opcode: u32,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let is_vop1 = (VOP1_START..VOP1_END).contains(&opcode);
    let is_vop2 = (VOP2_START..VOP2_END).contains(&opcode);
    if !(is_vop1 || is_vop2) || !vector::is_known(name) {
        return Ok(false);
    }
    let float = float_result(name);
    let saturating = matches!(name, "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32");
    if fields.clamp && !float && !saturating {
        return Err(Error::Untranslated(format!(
            "{name}: clamp on integer result"
        )));
    }
    let needs_literal = matches!(
        name,
        "v_madmk_f32" | "v_madak_f32" | "v_fmamk_f32" | "v_fmaak_f32"
    );
    if needs_literal && literal.is_none() {
        return Err(Error::Untranslated(format!("{name} en VOP3 sans littéral")));
    }
    let destination = fields.vdst as usize;
    let k = literal.unwrap_or(0);
    let select = if name == "v_cndmask_b32" {
        Some(wave.read_scalar(fields.sources[2], literal, wave.mask_width)?)
    } else {
        None
    };
    for lane in wave.active()? {
        let [a, b, _] = inputs(wave, fields, lane, literal, float_inputs(name))?;
        let b = if is_vop2 { b } else { 0 };
        let old = wave.v[destination][lane];
        let condition = select.is_some_and(|mask| mask >> lane & 1 == 1);
        let value = if fields.clamp && saturating {
            match name {
                "v_add_nc_u32" => a.saturating_add(b),
                "v_sub_nc_u32" => a.saturating_sub(b),
                _ => b.saturating_sub(a),
            }
        } else {
            vector::compute(name, a, b, old, k, condition).ok_or_else(|| {
                Error::Untranslated(format!("instruction non traduite : {name} (VOP3)"))
            })?
        };
        wave.v[destination][lane] = output(fields, u64::from(value), float);
    }
    Ok(true)
}

/// VOP3b carry forms: the carry of each lane goes to the SGPR pair `sdst`, the result to vdst.
pub fn carry(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if !vector::is_vop3_carry(name) {
        return Ok(false);
    }
    let carry_in = if name.ends_with("_ci_u32") {
        wave.read_scalar(fields.sources[2], literal, wave.mask_width)?
    } else {
        0
    };
    let mut carries = 0u64;
    for lane in wave.active()? {
        let [a, b, _] = inputs(wave, fields, lane, literal, false)?;
        let (result, carry) = vector::carry(name, u64::from(a), u64::from(b), carry_in >> lane & 1)
            .ok_or_else(|| Error::Operand(format!("retenue inconnue : {name}")))?;
        wave.v[fields.vdst as usize][lane] = if fields.clamp && carry {
            if name.starts_with("v_add") {
                u32::MAX
            } else {
                0
            }
        } else {
            (result & MASK32) as u32
        };
        carries |= u64::from(carry) << lane;
    }
    wave.write_mask(fields.sdst, carries)?;
    Ok(true)
}

/// Lane operations: readfirstlane, readlane, writelane and the two mbcnt counts.
pub fn lane(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let [src0, src1, _] = fields.sources;
    let lanes = wave.lanes as u64;
    match name {
        "v_readfirstlane_b32" => {
            let first = wave.active()?.first().copied().unwrap_or(0);
            let value = wave.read_lane(src0, first, literal)?;
            wave.write_scalar(fields.vdst, u64::from(value), 1)?;
        }
        "v_readlane_b32" => {
            let lane = (wave.read_scalar(src1, literal, 1)? & (lanes - 1)) as usize;
            let value = wave.read_lane(src0, lane, literal)?;
            wave.write_scalar(fields.vdst, u64::from(value), 1)?;
        }
        "v_writelane_b32" => {
            let lane = (wave.read_scalar(src1, literal, 1)? & (lanes - 1)) as usize;
            let value = wave.read_scalar(src0, literal, 1)? as u32;
            wave.v[fields.vdst as usize][lane] = value;
        }
        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
            for lane in wave.active()? {
                let below: u64 = (1u64 << lane) - 1;
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    below & MASK32
                } else {
                    below >> 32
                };
                let source = u64::from(wave.read_lane(src0, lane, literal)?);
                let count = (source & below).count_ones();
                let total = u64::from(count) + u64::from(wave.read_lane(src1, lane, literal)?);
                wave.v[fields.vdst as usize][lane] = (total & MASK32) as u32;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// v_mad_u64_u32 and v_mad_i64_i32: S0 × S1 + S2 (64 bits) into a VGPR pair; the carry goes to sdst.
pub fn mad64(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if name != "v_mad_u64_u32" && name != "v_mad_i64_i32" {
        return Ok(false);
    }
    if fields.clamp {
        return Err(Error::Untranslated(format!(
            "{name}: clamp on integer result"
        )));
    }
    let signed = name == "v_mad_i64_i32";
    if signed && fields.sdst != NULL {
        return Err(Error::Untranslated(
            "v_mad_i64_i32 : retenue signée dans SDST non vérifiée".into(),
        ));
    }
    let [src0, src1, src2] = fields.sources;
    let mut carries = 0u64;
    for lane in wave.active()? {
        let a = wave.read_lane(src0, lane, literal)?;
        let b = wave.read_lane(src1, lane, literal)?;
        let c = crate::compare::read64(wave, src2, lane, false)?;
        let total: i128 = if signed {
            i128::from(sign_extend(u64::from(a), 32)) * i128::from(sign_extend(u64::from(b), 32))
                + i128::from(sign_extend(c, 64))
        } else {
            i128::from(a) * i128::from(b) + i128::from(c)
        };
        let bits = total as u128;
        let destination = wave.vgpr_group(fields.vdst as usize, 2)?;
        wave.v[destination][lane] = (bits as u64 & MASK32) as u32;
        wave.v[destination + 1][lane] = ((bits as u64 >> 32) & 0xFFFF_FFFF) as u32;
        if !signed && bits >> 64 != 0 {
            carries |= 1u64 << lane;
        }
    }
    wave.write_mask(fields.sdst, carries)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_names_follow_the_model_rules() {
        assert!(float_result("v_add_f32"));
        assert!(!float_result("v_cvt_u32_f32"));
        assert!(float_inputs("v_cvt_f32_i32"));
        assert!(!float_result("v_add_nc_u32"));
    }

    #[test]
    fn modifiers_touch_only_the_sign_of_float_operands() {
        let fields = Fields::new(1 | 1 << 61 | 1 << 8);
        assert_eq!(modify(0x8000_0001, 0, &fields, true), 0x8000_0001);
        let absolute = Fields::new(1 << 8);
        assert_eq!(modify(0x8000_0001, 0, &absolute, true), 0x0000_0001);
        assert_eq!(modify(0x8000_0001, 0, &fields, false), 0x8000_0001);
    }

    #[test]
    fn literal_vop2_forms_in_vop3_require_a_literal() {
        let mut wave = Wave::new(32).expect("wave32");
        let fields = Fields::new(1 | (256u64 << 32) | (257u64 << 41));
        for (opcode, name) in [
            (0x120, "v_madmk_f32"),
            (0x121, "v_madak_f32"),
            (0x12C, "v_fmamk_f32"),
            (0x12D, "v_fmaak_f32"),
        ] {
            let err = operation(&mut wave, name, opcode, &fields, None).unwrap_err();
            assert!(matches!(err, Error::Untranslated(_)), "{name}: {err:?}");
            assert!(
                operation(&mut wave, name, opcode, &fields, Some(0x3F80_0000)).unwrap(),
                "{name} with literal executes"
            );
        }
    }
}
