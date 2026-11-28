//! Masked and quad SAD instructions of VOP3 (`v_msad_u8`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`,
//! `v_mqsad_u32_u8`): port of `emu/rdna2_exec_sad.py`.
//!
//! `v_msad_u8`: D = S2 + Σ |S0[i] - S1[i]| over the four bytes, skipping the bytes where the reference
//! S1[i] is zero. The quad forms take four 32-bit windows of the 64-bit S0 (at bit 0, 8, 16 and 24),
//! each one a SAD against S1 added to an accumulator of S2, 16 or 32 bits wide.
//! The VOP3 modifiers (abs, neg, clamp, omod) have no described meaning for these integers and are
//! refused, as are a literal or an inline float on a 64-bit source.

use crate::compare::read64;
use crate::error::Error;
use crate::vop3::Fields;
use crate::wave::{VGPR_FIRST, Wave};

const MASK32: u64 = 0xFFFF_FFFF;
/// Source codes below this are SGPRs (0-127) and inline integers (128-208): read as 64-bit values.
const INLINE_INTEGERS_END: u32 = 209;

/// The SAD instructions of this module, with their accumulator width (`None` for `v_msad_u8`).
pub fn operation(name: &str) -> Option<(bool, Option<u32>)> {
    match name {
        "v_msad_u8" => Some((true, None)),
        "v_qsad_pk_u16_u8" => Some((false, Some(16))),
        "v_mqsad_pk_u16_u8" => Some((true, Some(16))),
        "v_mqsad_u32_u8" => Some((true, Some(32))),
        _ => None,
    }
}

/// Σ |a[i] - b[i]| over four bytes; `masked`: the bytes where `b[i]` is zero do not count.
pub fn sad(a: u64, b: u64, masked: bool) -> u64 {
    let mut total = 0;
    for shift in [0, 8, 16, 24] {
        let (x, y) = ((a >> shift) & 0xFF, (b >> shift) & 0xFF);
        if !(masked && y == 0) {
            total += x.abs_diff(y);
        }
    }
    total
}

/// The packed results of a quad SAD: four windows of S0 against S1, each added to its accumulator.
pub fn quad(name: &str, s0: u64, s1: u64, s2: u128) -> Option<u128> {
    let (masked, width) = operation(name)?;
    let width = width?;
    let mask = (1u128 << width) - 1;
    let mut result = 0u128;
    for k in 0..4 {
        let window = (s0 >> (8 * k)) & MASK32;
        let accumulator = (s2 >> (width * k)) & mask;
        let total = (u128::from(sad(window, s1, masked)) + accumulator) & mask;
        result |= total << (width * k);
    }
    Some(result)
}

/// The reason a SAD form is not executed, if any (the Python `refusal`).
fn refusal(name: &str, fields: &Fields) -> Option<String> {
    let present: Vec<&str> = [
        ("abs", fields.abs != 0),
        ("neg", fields.neg != 0),
        ("clamp", fields.clamp),
        ("omod", fields.omod != 0),
    ]
    .iter()
    .filter(|(_, set)| *set)
    .map(|(label, _)| *label)
    .collect();
    if !present.is_empty() {
        return Some(format!(
            "{name} avec {} : sens non documenté",
            present.join(", ")
        ));
    }
    let [src0, _, src2] = fields.sources;
    if name == "v_mqsad_u32_u8" && src2 < VGPR_FIRST {
        return Some(format!("{name} : S2 (128 bits) doit être un VGPR"));
    }
    if name != "v_msad_u8" {
        for code in [src0, src2] {
            if !((VGPR_FIRST..).contains(&code) || code < INLINE_INTEGERS_END) {
                return Some(format!(
                    "{name} : littéral ou constante flottante sur une source de 64 bits"
                ));
            }
        }
    }
    None
}

/// S2 of `words` 32-bit words: four VGPRs, or a 64-bit source (register pair or inline constant).
fn wide_source(wave: &Wave, code: u32, lane: usize, words: usize) -> Result<u128, Error> {
    if code >= VGPR_FIRST {
        let base = wave.vgpr_group((code - VGPR_FIRST) as usize, words)?;
        let mut value = 0u128;
        for i in 0..words {
            value |= u128::from(wave.v[base + i][lane]) << (32 * i);
        }
        return Ok(value);
    }
    Ok(u128::from(read64(wave, code, lane, false)?))
}

/// Executes a SAD instruction for every active lane. `Ok(false)` when `name` is not one of them.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if operation(name).is_none() {
        return Ok(false);
    }
    if let Some(message) = refusal(name, fields) {
        return Err(Error::Untranslated(message));
    }
    let [src0, src1, src2] = fields.sources;
    let words = if name == "v_mqsad_u32_u8" { 4 } else { 2 };
    for lane in wave.active()? {
        let b = u64::from(wave.read_lane(src1, lane, literal)?);
        if name == "v_msad_u8" {
            let a = u64::from(wave.read_lane(src0, lane, literal)?);
            let c = u64::from(wave.read_lane(src2, lane, literal)?);
            wave.v[fields.vdst as usize][lane] = ((sad(a, b, true) + c) & MASK32) as u32;
            continue;
        }
        let s0 = read64(wave, src0, lane, false)?;
        let s2 = wide_source(wave, src2, lane, words)?;
        let result = quad(name, s0, b, s2).unwrap_or(0);
        let destination = wave.vgpr_group(fields.vdst as usize, words)?;
        for i in 0..words {
            wave.v[destination + i][lane] = ((result >> (32 * i)) & u128::from(MASK32)) as u32;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_sad_skips_the_reference_zero_bytes() {
        assert_eq!(sad(0x0102_0304, 0x0102_0300, false), 4);
        assert_eq!(sad(0x0102_0304, 0x0102_0300, true), 0);
        assert_eq!(sad(0x0A0B_0C0D, 0x0102_0304, false), 9 + 9 + 9 + 9);
    }

    #[test]
    fn quad_of_zeros_is_zero_and_each_window_adds_its_accumulator() {
        assert_eq!(quad("v_qsad_pk_u16_u8", 0, 0, 0), Some(0));
        // Every window is 0 against 0 (SAD 0): only the accumulators remain, 16 bits each.
        let accumulators = 1u128 | 2 << 16 | 3 << 32 | 4 << 48;
        assert_eq!(
            quad("v_qsad_pk_u16_u8", 0, 0, accumulators),
            Some(accumulators)
        );
    }

    #[test]
    fn a_modifier_or_a_literal_is_refused() {
        let fields = Fields::new(1 << 15);
        assert!(refusal("v_msad_u8", &fields).is_some());
    }
}
