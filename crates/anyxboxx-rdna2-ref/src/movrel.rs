//! Registers indexed by M0, and register swaps: `v_movrels_b32`, `v_movreld_b32`, `v_movrelsd_b32`,
//! `v_movrelsd_2_b32`, `v_swap_b32`, `v_swaprel_b32`, and the scalar `s_movrels`/`s_movreld` forms.
//! Port of `emu/rdna2_exec_movrel.py`.
//!
//! The index is the register number of the instruction plus M0 (plus M0[9:0] for the source and M0[25:16]
//! for the destination of the `_2` forms and v_swaprel). Indexed sources must be registers, an index outside
//! 256 VGPRs or s0 to s105, and an odd 64-bit index are refused, as the Python does.

use crate::error::Error;
use crate::scalar::Fields;
use crate::wave::{M0, Wave};

const VGPR_FIRST: u32 = 256;
const LITERAL: u32 = 255;
const VGPR_LAST: u32 = 255;
const LAST_SGPR: u32 = 105;
const FIELD_MASK: u32 = 0x3FF;
const MASK32: u64 = 0xFFFF_FFFF;

/// The vector forms (VOP1 and VOP3 encodings).
pub fn is_vector(name: &str) -> bool {
    matches!(
        name,
        "v_movrels_b32"
            | "v_movreld_b32"
            | "v_movrelsd_b32"
            | "v_movrelsd_2_b32"
            | "v_swap_b32"
            | "v_swaprel_b32"
    )
}

/// The scalar forms (SOP1).
pub fn is_scalar(name: &str) -> bool {
    matches!(
        name,
        "s_movrels_b32" | "s_movrels_b64" | "s_movreld_b32" | "s_movreld_b64" | "s_movrelsd_2_b32"
    )
}

/// (offset of the source, offset of the destination) added to the register numbers.
fn offsets(name: &str, m0: u32) -> (u32, u32) {
    if name.ends_with("_2_b32") || name == "v_swaprel_b32" {
        return (m0 & FIELD_MASK, (m0 >> 16) & FIELD_MASK);
    }
    let source = matches!(
        name,
        "v_movrels_b32" | "v_movrelsd_b32" | "s_movrels_b32" | "s_movrels_b64"
    );
    let destination = matches!(
        name,
        "v_movreld_b32" | "v_movrelsd_b32" | "s_movreld_b32" | "s_movreld_b64"
    );
    (
        if source { m0 } else { 0 },
        if destination { m0 } else { 0 },
    )
}

fn refused(message: String) -> Error {
    Error::Untranslated(message)
}

/// The register `register + offset`, which must stay within `last`; a 64-bit one must be even.
///
/// The sum is taken in 64 bits, like the Python's unbounded integers: M0 is a full 32-bit value, so an
/// offset such as 0xFFFF_FFFD added to register 47 is 0x1_0000_002C, outside the model, and not register
/// 44 as a wrapping 32-bit sum would make it (nor a panic, in a build that checks overflow).
fn check(name: &str, register: u32, offset: u32, last: u32, width: u32) -> Result<u32, Error> {
    let index = u64::from(register) + u64::from(offset);
    if index + u64::from(width) - 1 > u64::from(last) || (width == 2 && index % 2 == 1) {
        return Err(refused(format!(
            "{name} : registre indexé {index} hors du modèle"
        )));
    }
    // In range: at most `last`, which is a u32.
    Ok(index as u32)
}

/// A vector instruction of the VOP1 or VOP3 encodings, for every active lane.
pub fn vector(
    wave: &mut Wave,
    name: &str,
    vdst: u32,
    src0: u32,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if !is_vector(name) {
        return Ok(false);
    }
    let (source_offset, destination_offset) = offsets(name, wave.s[M0 as usize]);
    let destination = check(name, vdst, destination_offset, VGPR_LAST, 1)? as usize;
    if name == "v_movreld_b32" {
        for lane in wave.active()? {
            wave.v[destination][lane] = wave.read_lane(src0, lane, literal)?;
        }
        return Ok(true);
    }
    if src0 < VGPR_FIRST {
        return Err(refused(format!(
            "{name} : la source indexée doit être un VGPR"
        )));
    }
    let source = check(name, src0 - VGPR_FIRST, source_offset, VGPR_LAST, 1)? as usize;
    let swap = matches!(name, "v_swap_b32" | "v_swaprel_b32");
    for lane in wave.active()? {
        let value = wave.v[source][lane];
        if swap {
            wave.v[source][lane] = wave.v[destination][lane];
        }
        wave.v[destination][lane] = value;
    }
    Ok(true)
}

/// A scalar instruction of the SOP1 encoding. `Ok(false)` when `name` is not one of them.
pub fn scalar(
    wave: &mut Wave,
    name: &str,
    fields: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if !is_scalar(name) {
        return Ok(false);
    }
    let width: u32 = if name.ends_with("_b64") { 2 } else { 1 };
    let (source_offset, destination_offset) = offsets(name, wave.s[M0 as usize]);
    let (sdst, ssrc0) = (fields.sdst, fields.ssrc0);
    let value = if matches!(name, "s_movreld_b32" | "s_movreld_b64") {
        if width == 2 && ssrc0 == LITERAL {
            return Err(refused(format!("{name} : littéral sur une source 64 bits")));
        }
        wave.read_scalar(ssrc0, literal, width as usize)?
    } else {
        if ssrc0 > LAST_SGPR {
            return Err(refused(format!(
                "{name} : la source indexée doit être un SGPR"
            )));
        }
        let index = check(name, ssrc0, source_offset, LAST_SGPR, width)? as usize;
        let low = u64::from(wave.s[index]);
        if width == 2 {
            low | u64::from(wave.s[index + 1]) << 32
        } else {
            low
        }
    };
    if matches!(name, "s_movrels_b32" | "s_movrels_b64") {
        wave.write_scalar(sdst, value, width as usize)?;
        return Ok(true);
    }
    if sdst > LAST_SGPR {
        return Err(refused(format!(
            "{name} : la destination indexée doit être un SGPR"
        )));
    }
    let target = check(name, sdst, destination_offset, LAST_SGPR, width)? as usize;
    wave.s[target] = (value & MASK32) as u32;
    if width == 2 {
        wave.s[target + 1] = ((value >> 32) & MASK32) as u32;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offsets_come_from_m0_by_form() {
        assert_eq!(offsets("v_movrels_b32", 5), (5, 0));
        assert_eq!(offsets("v_movreld_b32", 5), (0, 5));
        assert_eq!(offsets("v_swaprel_b32", 0x0003_0002), (2, 3));
    }

    #[test]
    fn an_odd_64_bit_index_is_refused() {
        assert!(check("s_movrels_b64", 3, 0, 105, 2).is_err());
        assert_eq!(check("s_movrels_b64", 4, 0, 105, 2).unwrap(), 4);
        assert_eq!(check("s_movrels_b64", 1, 3, 105, 2).unwrap(), 4);
    }

    #[test]
    fn an_index_past_32_bits_is_refused_not_wrapped() {
        // 47 + 0xFFFF_FFFD is 0x1_0000_002C; a wrapping sum would give register 44.
        assert!(check("s_movrels_b32", 47, 0xFFFF_FFFD, 105, 1).is_err());
        assert!(check("v_movreld_b32", 3, 0xFFFF_FFFF, 255, 1).is_err());
        // The pair's second register is counted in 64 bits too.
        assert!(check("s_movrels_b64", 1, 0xFFFF_FFFF, 105, 2).is_err());
        assert!(check("s_movrels_b64", 0, 0xFFFF_FFFF, 105, 2).is_err());
        assert_eq!(check("s_movrels_b32", 105, 0, 105, 1).unwrap(), 105);
        assert!(check("s_movrels_b32", 105, 1, 105, 1).is_err());
    }
}
