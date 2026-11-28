//! Scalar half-precision (f16) arithmetic, rounding and conversions of VOP1, VOP2 and VOP3, ported from
//! `emu/rdna2_exec_f16.py` (excluding the seven transcendentals, which have their own module).

use crate::error::Error;
use crate::float::bits_float;
use crate::half::{
    HALF_MASK, SourceKind, half_clamp, half_fract, half_frexp, half_from_f64, half_max, half_min,
    half_round, half_to_f32_bits, half_to_f64, half_toward_zero, modify_source, pow2, read_source,
};
use crate::vop3::Fields;
use crate::wave::{VGPR_FIRST, Wave};

/// Source codes of the DPP8, DPP8_FI, SDWA and DPP16 forms: refused for f16.
const EXTENSION_CODES: [u32; 4] = [0xE9, 0xEA, 0xF9, 0xFA];

type F16Op = fn(u32, u32, u32, u32, u32) -> u32;

const H: &[SourceKind] = &[SourceKind::Half];
const HH: &[SourceKind] = &[SourceKind::Half, SourceKind::Half];
const HHH: &[SourceKind] = &[SourceKind::Half, SourceKind::Half, SourceKind::Half];
const HI: &[SourceKind] = &[SourceKind::Half, SourceKind::Int16];
const W: &[SourceKind] = &[SourceKind::Word];
const WW: &[SourceKind] = &[SourceKind::Word, SourceKind::Word];

fn h(bits: u32) -> f64 {
    half_to_f64(bits)
}

fn ldexp_f16(a: u32, b: u32) -> u32 {
    let exp = i32::from(b as u16 as i16).clamp(-64, 64);
    half_from_f64(h(a) * pow2(exp))
}

fn spec(name: &str) -> Option<(&'static [SourceKind], F16Op)> {
    Some(match name {
        "v_add_f16" => (HH, |a, b, _, _, _| half_from_f64(h(a) + h(b))),
        "v_sub_f16" => (HH, |a, b, _, _, _| half_from_f64(h(a) - h(b))),
        "v_subrev_f16" => (HH, |a, b, _, _, _| half_from_f64(h(b) - h(a))),
        "v_mul_f16" => (HH, |a, b, _, _, _| half_from_f64(h(a) * h(b))),
        "v_fmac_f16" => (HH, |a, b, _, d, _| half_from_f64(h(a) * h(b) + h(d))),
        "v_fmamk_f16" => (HH, |a, b, _, _, k| half_from_f64(h(a) * h(k) + h(b))),
        "v_fmaak_f16" => (HH, |a, b, _, _, k| half_from_f64(h(a) * h(b) + h(k))),
        "v_fma_f16" => (HHH, |a, b, c, _, _| half_from_f64(h(a) * h(b) + h(c))),
        "v_max_f16" => (HH, |a, b, _, _, _| half_max(a, b)),
        "v_min_f16" => (HH, |a, b, _, _, _| half_min(a, b)),
        "v_ldexp_f16" => (HI, |a, b, _, _, _| ldexp_f16(a, b)),
        "v_floor_f16" => (H, |a, _, _, _, _| half_round(a, f64::floor)),
        "v_ceil_f16" => (H, |a, _, _, _, _| half_round(a, f64::ceil)),
        "v_trunc_f16" => (H, |a, _, _, _, _| half_round(a, f64::trunc)),
        "v_rndne_f16" => (H, |a, _, _, _, _| half_round(a, f64::round_ties_even)),
        "v_cvt_pkrtz_f16_f32" => (WW, |a, b, _, _, _| {
            half_toward_zero(bits_float(b)) << 16 | half_toward_zero(bits_float(a))
        }),
        "v_cvt_f16_f32" => (W, |a, _, _, _, _| half_from_f64(bits_float(a))),
        "v_cvt_f32_f16" => (H, |a, _, _, _, _| half_to_f32_bits(a)),
        "v_frexp_mant_f16" => (H, |a, _, _, _, _| half_frexp(a).0),
        "v_frexp_exp_i16_f16" => (H, |a, _, _, _, _| (half_frexp(a).1 as u32) & HALF_MASK),
        "v_fract_f16" => (H, |a, _, _, _, _| half_fract(a)),
        _ => return None,
    })
}

/// Whether `name` is one of the scalar f16 arithmetic, rounding or conversion instructions handled here.
pub fn supports(name: &str) -> bool {
    spec(name).is_some()
}

fn is_word_result(name: &str) -> bool {
    matches!(name, "v_cvt_f32_f16" | "v_cvt_pkrtz_f16_f32")
}

fn is_integer_result(name: &str) -> bool {
    name == "v_frexp_exp_i16_f16"
}

/// Executes a VOP1 or VOP2 scalar f16 instruction (`word` is the 32-bit encoding).
pub fn execute(
    wave: &mut Wave,
    name: &str,
    family: &str,
    word: u32,
    literal: Option<u32>,
) -> Result<(), Error> {
    let src0 = word & 0x1FF;
    if EXTENSION_CODES.contains(&src0) {
        return Err(Error::Untranslated(format!(
            "{name}: SDWA/DPP form (src0 = 0x{src0:X}) not supported in f16"
        )));
    }
    let (kinds, operation) = spec(name).ok_or_else(|| Error::Untranslated(name.to_string()))?;
    let codes: [u32; 2] = if family == "VOP2" {
        [src0, VGPR_FIRST + ((word >> 9) & 0xFF)]
    } else {
        [src0, 0]
    };
    let vdst = ((word >> 17) & 0xFF) as usize;
    let k = literal.unwrap_or(0) & HALF_MASK;
    let word_out = is_word_result(name);
    for lane in wave.active()? {
        let mut args = [0u32; 2];
        for (index, &kind) in kinds.iter().enumerate() {
            args[index] = read_source(wave, name, kind, codes[index], lane, literal, false)?;
        }
        let old = wave.v[vdst][lane];
        let result = operation(args[0], args[1], 0, old & HALF_MASK, k);
        wave.v[vdst][lane] = if word_out {
            result
        } else {
            (old & 0xFFFF_0000) | result
        };
    }
    Ok(())
}

fn check_vop3(
    name: &str,
    kinds: &[SourceKind],
    fields: &Fields,
    op_sel: u32,
    literal: Option<u32>,
) -> Result<(), Error> {
    let word_out = is_word_result(name);
    if word_out && (fields.clamp || fields.omod != 0) {
        return Err(Error::Untranslated(format!(
            "{name}: clamp or omod on an f32 result not supported"
        )));
    }
    if fields.omod != 0 {
        return Err(Error::Untranslated(format!(
            "{name}: omod on an f16 result not supported"
        )));
    }
    if fields.clamp && is_integer_result(name) {
        return Err(Error::Untranslated(format!(
            "{name}: clamp on an integer result not measured"
        )));
    }
    if op_sel != 0 && name == "v_fmac_f16" {
        return Err(Error::Untranslated(format!(
            "{name}: op_sel not supported in VOP3"
        )));
    }
    if (op_sel >> 3) & 1 != 0 && word_out {
        return Err(Error::Untranslated(format!(
            "{name}: op_sel on a 32-bit destination not supported"
        )));
    }
    if matches!(name, "v_fmamk_f16" | "v_fmaak_f16") && literal.is_none() {
        return Err(Error::Untranslated(format!(
            "{name} (VOP3): missing literal constant K"
        )));
    }
    for (index, &kind) in kinds.iter().enumerate() {
        if kind == SourceKind::Word && (op_sel >> index) & 1 != 0 {
            return Err(Error::Untranslated(format!(
                "{name}: op_sel on a 32-bit source not supported"
            )));
        }
        if kind == SourceKind::Int16 && ((fields.abs | fields.neg) >> index) & 1 != 0 {
            return Err(Error::Untranslated(format!(
                "{name}: abs/neg on an integer source not supported"
            )));
        }
    }
    Ok(())
}

/// Executes a VOP3 scalar f16 instruction (`word` is the 64-bit encoding).
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    word: u64,
    literal: Option<u32>,
) -> Result<(), Error> {
    let (kinds, operation) = spec(name).ok_or_else(|| Error::Untranslated(name.to_string()))?;
    let fields = Fields::new(word);
    let op_sel = ((word >> 11) & 0xF) as u32;
    check_vop3(name, kinds, &fields, op_sel, literal)?;
    let high_dst = (op_sel >> 3) & 1 != 0;
    let vdst = fields.vdst as usize;
    let k = literal.unwrap_or(0) & HALF_MASK;
    let word_out = is_word_result(name);
    for lane in wave.active()? {
        let mut args = [0u32; 3];
        for (index, &kind) in kinds.iter().enumerate() {
            let high_src = (op_sel >> index) & 1 != 0;
            let raw = read_source(
                wave,
                name,
                kind,
                fields.sources[index],
                lane,
                literal,
                high_src,
            )?;
            args[index] = modify_source(raw, kind, fields.abs, fields.neg, index);
        }
        let old = wave.v[vdst][lane];
        let mut result = operation(args[0], args[1], args[2], old & HALF_MASK, k);
        if fields.clamp {
            result = half_clamp(result);
        }
        wave.v[vdst][lane] = if word_out {
            result
        } else if high_dst {
            (old & HALF_MASK) | (result << 16)
        } else {
            (old & 0xFFFF_0000) | result
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_rounding_and_conversions_follow_the_model() {
        let (_, add) = spec("v_add_f16").unwrap();
        assert_eq!(add(0x3C00, 0x4000, 0, 0, 0), 0x4200);
        let (_, fma) = spec("v_fma_f16").unwrap();
        assert_eq!(fma(0x3E00, 0x4000, 0x3800, 0, 0), 0x4300);
        let (_, pkrtz) = spec("v_cvt_pkrtz_f16_f32").unwrap();
        // 70000.0f32 -> +65504 (0x7BFF), -1.5f32 -> -1.5 (0xBE00).
        assert_eq!(
            pkrtz(70000.0f32.to_bits(), (-1.5f32).to_bits(), 0, 0, 0),
            0xBE00_7BFF
        );
        let (_, fract) = spec("v_fract_f16").unwrap();
        assert_eq!(fract(0xBC00, 0, 0, 0, 0), 0x0000);
        assert_eq!(fract(0xBBFF, 0, 0, 0, 0), 0x1000);
        assert_eq!(fract(0x7C00, 0, 0, 0, 0), 0x7E00);
    }

    #[test]
    fn vop3_refuses_unsupported_modifiers_before_mutating_lanes() {
        let mut wave = Wave::new(32).unwrap();
        wave.v[1][0] = 0xAAAA_3C00;
        let word_omod = (0x35u64 << 26)
            | (0x208u64 << 16)
            | 1
            | ((257u64 | (257u64 << 9)) << 32)
            | (1u64 << 59);
        assert!(matches!(
            execute_vop3(&mut wave, "v_fma_f16", word_omod, None),
            Err(Error::Untranslated(_))
        ));
        assert_eq!(wave.v[1][0], 0xAAAA_3C00);
    }
}
