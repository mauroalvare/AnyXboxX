//! 16-bit integer VOP3 instructions, ported from `emu/rdna2_exec_int16.py` (`execute_vop3`, 26 encodings):
//! the VOP3-only 16-bit operations (`v_add/sub_nc_{u16,i16}`, `v_mul_lo_u16`, `v_lshlrev_b16`,
//! `v_lshrrev_b16`, `v_ashrrev_i16`, `v_max/min_{u16,i16}`, `v_mad_{u16,i16}`, `v_min3/max3/med3_{u16,i16}`),
//! the 16 x 16 + 32 multiply-adds `v_mad_u32_u16` and `v_mad_i32_i16`, the saturating packs
//! `v_cvt_pk_u16_u32` and `v_cvt_pk_i16_i32`, and the signed 32-bit `v_add_nc_i32` and `v_sub_nc_i32`.
//! The packed VOP3P forms of the same Python module are in `vop3p.rs`.
//!
//! Semantics, as in the Python module (whose docstring cites AMD's RDNA 2 ISA, LLVM and Mesa):
//! - 16-bit operations: op_sel[i] reads S_i[31:16] instead of S_i[15:0], op_sel[3] writes D[31:16] instead
//!   of D[15:0], and the other half of D is kept. The result wraps to 16 bits; with clamp, add, sub and mad
//!   saturate the exact value to [0, 0xFFFF] (u16) or [-32768, 32767] (i16).
//! - `v_mad_u32_u16` / `v_mad_i32_i16`: D = S0.half * S1.half + S2, the halves zero- or sign-extended and
//!   picked by op_sel[0-1]; clamp saturates the 32-bit sum to the type's range.
//! - `v_cvt_pk_u16_u32` / `v_cvt_pk_i16_i32`: D = {sat16(S1), sat16(S0)}.
//! - `v_add_nc_i32` / `v_sub_nc_i32`: 32-bit wrap; clamp saturates to [0x80000000, 0x7FFFFFFF].
//!
//! Refused (Untranslated), as in the Python: abs, neg and omod (float modifiers); clamp where the model does
//! not implement it (`v_mul_lo_u16`, the shifts, min/max, min3/max3/med3, the packs); op_sel[2] and op_sel[3]
//! on the 32-bit multiply-adds; op_sel on the packs and on `v_add/sub_nc_i32`. Only the sources an operation
//! uses are read: the select bit of an unused third source is ignored.

use crate::error::Error;
use crate::vop3::Fields;
use crate::wave::Wave;

const HALF_MASK: u32 = 0xFFFF;
const HALF_BITS: u32 = 16;
const HIGH_HALF: u32 = 0xFFFF_0000;
const MASK32: i64 = 0xFFFF_FFFF;
const SHIFT_MASK: i64 = 15;
const OP_SEL_SHIFT: u32 = 11;
const OP_SEL_MASK: u64 = 0xF;
/// op_sel[3]: the result goes to the high half of D.
const DESTINATION_HIGH: u32 = 1 << 3;
/// op_sel[2] and op_sel[3]: select bits on the 32-bit S2 and D of the 32-bit multiply-adds.
const WIDE_SELECTS: u32 = 0b1100;
const U16: (i64, i64) = (0, 0xFFFF);
const I16: (i64, i64) = (-0x8000, 0x7FFF);
const U32: (i64, i64) = (0, MASK32);
const I32: (i64, i64) = (i32::MIN as i64, i32::MAX as i64);

/// An operation on three zero-extended 16-bit halves, giving the exact integer result.
type HalfOp = fn(i64, i64, i64) -> i64;

#[derive(Debug, Clone, Copy)]
enum Form {
    /// A 16-bit operation, the number of sources it reads, and the range clamp saturates to (None: clamp refused).
    Half {
        op: HalfOp,
        sources: usize,
        saturate: Option<(i64, i64)>,
    },
    /// `v_mad_u32_u16` (false) or `v_mad_i32_i16` (true).
    Mad32 { signed: bool },
    /// `v_cvt_pk_u16_u32` or `v_cvt_pk_i16_i32`: the range of each half, and whether the sources are signed.
    Pack { bounds: (i64, i64), signed: bool },
    /// `v_add_nc_i32` (+1) or `v_sub_nc_i32` (-1): the sign of S1.
    AddSub32 { sign: i64 },
}

fn s16(value: i64) -> i64 {
    i64::from(value as u16 as i16)
}

fn s32(value: u32) -> i64 {
    i64::from(value as i32)
}

fn median(a: i64, b: i64, c: i64) -> i64 {
    a.min(b).max(a.max(b).min(c))
}

fn half(op: HalfOp, sources: usize, saturate: Option<(i64, i64)>) -> Form {
    Form::Half {
        op,
        sources,
        saturate,
    }
}

fn form(name: &str) -> Option<Form> {
    Some(match name {
        "v_add_nc_u16" => half(|a, b, _| a + b, 2, Some(U16)),
        "v_sub_nc_u16" => half(|a, b, _| a - b, 2, Some(U16)),
        "v_add_nc_i16" => half(|a, b, _| s16(a) + s16(b), 2, Some(I16)),
        "v_sub_nc_i16" => half(|a, b, _| s16(a) - s16(b), 2, Some(I16)),
        "v_mad_u16" => half(|a, b, c| a * b + c, 3, Some(U16)),
        "v_mad_i16" => half(|a, b, c| s16(a) * s16(b) + s16(c), 3, Some(I16)),
        "v_mul_lo_u16" => half(|a, b, _| a * b, 2, None),
        "v_lshlrev_b16" => half(|a, b, _| b << (a & SHIFT_MASK), 2, None),
        "v_lshrrev_b16" => half(|a, b, _| b >> (a & SHIFT_MASK), 2, None),
        "v_ashrrev_i16" => half(|a, b, _| s16(b) >> (a & SHIFT_MASK), 2, None),
        "v_max_u16" => half(|a, b, _| a.max(b), 2, None),
        "v_max_i16" => half(|a, b, _| s16(a).max(s16(b)), 2, None),
        "v_min_u16" => half(|a, b, _| a.min(b), 2, None),
        "v_min_i16" => half(|a, b, _| s16(a).min(s16(b)), 2, None),
        "v_min3_u16" => half(|a, b, c| a.min(b).min(c), 3, None),
        "v_min3_i16" => half(|a, b, c| s16(a).min(s16(b)).min(s16(c)), 3, None),
        "v_max3_u16" => half(|a, b, c| a.max(b).max(c), 3, None),
        "v_max3_i16" => half(|a, b, c| s16(a).max(s16(b)).max(s16(c)), 3, None),
        "v_med3_u16" => half(median, 3, None),
        "v_med3_i16" => half(|a, b, c| median(s16(a), s16(b), s16(c)), 3, None),
        "v_mad_u32_u16" => Form::Mad32 { signed: false },
        "v_mad_i32_i16" => Form::Mad32 { signed: true },
        "v_cvt_pk_u16_u32" => Form::Pack {
            bounds: U16,
            signed: false,
        },
        "v_cvt_pk_i16_i32" => Form::Pack {
            bounds: I16,
            signed: true,
        },
        "v_add_nc_i32" => Form::AddSub32 { sign: 1 },
        "v_sub_nc_i32" => Form::AddSub32 { sign: -1 },
        _ => return None,
    })
}

/// Whether `name` is one of the 26 VOP3 encodings of this module.
pub fn supports(name: &str) -> bool {
    form(name).is_some()
}

fn source_count(form: Form) -> usize {
    match form {
        Form::Half { sources, .. } => sources,
        Form::Mad32 { .. } => 3,
        Form::Pack { .. } | Form::AddSub32 { .. } => 2,
    }
}

/// The reason the model refuses this form (the Python `vop3_refusal`), or None.
fn refusal(name: &str, form: Form, fields: &Fields, op_sel: u32) -> Option<String> {
    let present: Vec<&str> = [
        ("abs", fields.abs != 0),
        ("neg", fields.neg != 0),
        ("omod", fields.omod != 0),
    ]
    .iter()
    .filter(|(_, set)| *set)
    .map(|(label, _)| *label)
    .collect();
    if !present.is_empty() {
        return Some(format!(
            "{name} with {}: float modifiers on integer sources, not defined",
            present.join(", ")
        ));
    }
    match form {
        Form::Pack { .. } | Form::AddSub32 { .. } if op_sel != 0 => {
            Some(format!("{name} with op_sel: 32-bit sources, not defined"))
        }
        Form::Pack { .. } if fields.clamp => Some(format!("{name} with clamp: not implemented")),
        Form::Mad32 { .. } if op_sel & WIDE_SELECTS != 0 => Some(format!(
            "{name} with op_sel[2]/op_sel[3]: select bits on a 32-bit operand, not defined"
        )),
        Form::Half { saturate: None, .. } if fields.clamp => {
            Some(format!("{name} with clamp: not implemented"))
        }
        _ => None,
    }
}

/// The half of `value` that select bit `index` of `op_sel` names, zero-extended.
fn selected_half(value: u32, op_sel: u32, index: usize) -> i64 {
    i64::from(if op_sel >> index & 1 != 0 {
        value >> HALF_BITS
    } else {
        value & HALF_MASK
    })
}

/// The new value of D in one lane: `values` are the sources the form reads, `old` is D before.
fn lane_result(form: Form, op_sel: u32, clamp: bool, values: &[u32; 3], old: u32) -> u32 {
    match form {
        Form::Half { op, saturate, .. } => {
            let [a, b, c] = [0, 1, 2].map(|index| selected_half(values[index], op_sel, index));
            let mut value = op(a, b, c);
            if let (true, Some((low, high))) = (clamp, saturate) {
                value = value.clamp(low, high);
            }
            let result = (value & i64::from(HALF_MASK)) as u32;
            if op_sel & DESTINATION_HIGH != 0 {
                old & HALF_MASK | result << HALF_BITS
            } else {
                old & HIGH_HALF | result
            }
        }
        Form::Mad32 { signed } => {
            let (a, b) = (
                selected_half(values[0], op_sel, 0),
                selected_half(values[1], op_sel, 1),
            );
            let total = if signed {
                s16(a) * s16(b) + s32(values[2])
            } else {
                a * b + i64::from(values[2])
            };
            let (low, high) = if signed { I32 } else { U32 };
            let total = if clamp { total.clamp(low, high) } else { total };
            (total & MASK32) as u32
        }
        Form::Pack { bounds, signed } => {
            let read = |value: u32| if signed { s32(value) } else { i64::from(value) };
            let [a, b] = [values[0], values[1]]
                .map(|value| (read(value).clamp(bounds.0, bounds.1) & i64::from(HALF_MASK)) as u32);
            a | b << HALF_BITS
        }
        Form::AddSub32 { sign } => {
            let total = s32(values[0]) + sign * s32(values[1]);
            let total = if clamp {
                total.clamp(I32.0, I32.1)
            } else {
                total
            };
            (total & MASK32) as u32
        }
    }
}

/// Executes one of this module's VOP3 instructions on every active lane (`word`: the two VOP3 words).
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    word: u64,
    literal: Option<u32>,
) -> Result<(), Error> {
    let form = form(name).ok_or_else(|| Error::Untranslated(name.to_string()))?;
    let fields = Fields::new(word);
    let op_sel = ((word >> OP_SEL_SHIFT) & OP_SEL_MASK) as u32;
    if let Some(message) = refusal(name, form, &fields, op_sel) {
        return Err(Error::Untranslated(message));
    }
    let count = source_count(form);
    let vdst = fields.vdst as usize;
    for lane in wave.active()? {
        let mut values = [0u32; 3];
        for (index, value) in values.iter_mut().enumerate().take(count) {
            *value = wave.read_lane(fields.sources[index], lane, literal)?;
        }
        let old = wave.v[vdst][lane];
        wave.v[vdst][lane] = lane_result(form, op_sel, fields.clamp, &values, old);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(name: &str, op_sel: u32, clamp: bool, values: [u32; 3], old: u32) -> u32 {
        lane_result(form(name).unwrap(), op_sel, clamp, &values, old)
    }

    #[test]
    fn half_operations_wrap_and_saturate_only_with_clamp() {
        assert_eq!(run("v_add_nc_u16", 0, false, [0xFFFF, 2, 0], 0), 1);
        assert_eq!(run("v_add_nc_u16", 0, true, [0xFFFF, 2, 0], 0), 0xFFFF);
        assert_eq!(run("v_sub_nc_u16", 0, true, [1, 2, 0], 0), 0);
        assert_eq!(run("v_add_nc_i16", 0, true, [0x7FFF, 1, 0], 0), 0x7FFF);
        assert_eq!(run("v_sub_nc_i16", 0, true, [0x8000, 1, 0], 0), 0x8000);
        assert_eq!(run("v_mad_i16", 0, false, [0xFFFF, 0xFFFF, 0xFFFF], 0), 0);
    }

    #[test]
    fn op_sel_picks_source_halves_and_the_destination_half_keeps_the_other() {
        // S0 high half 3, S1 low half 4: 7 in the high half of D, its low half kept.
        assert_eq!(
            run(
                "v_add_nc_u16",
                0b1001,
                false,
                [0x0003_0000, 0x0005_0004, 0],
                0xAAAA_BBBB
            ),
            0x0007_BBBB
        );
        assert_eq!(
            run("v_add_nc_u16", 0, false, [1, 2, 0], 0xAAAA_BBBB),
            0xAAAA_0003
        );
    }

    #[test]
    fn signed_halves_compare_and_shift_as_signed_values() {
        assert_eq!(run("v_max_i16", 0, false, [0xFFFF, 1, 0], 0), 1);
        assert_eq!(run("v_max_u16", 0, false, [0xFFFF, 1, 0], 0), 0xFFFF);
        assert_eq!(run("v_ashrrev_i16", 0, false, [4, 0x8000, 0], 0), 0xF800);
        assert_eq!(run("v_lshrrev_b16", 0, false, [0x14, 0x8000, 0], 0), 0x0800);
        assert_eq!(run("v_med3_i16", 0, false, [0xFFFF, 5, 0x8000], 0), 0xFFFF);
        assert_eq!(run("v_med3_u16", 0, false, [0xFFFF, 5, 0x8000], 0), 0x8000);
    }

    #[test]
    fn the_32_bit_forms_wrap_or_saturate() {
        assert_eq!(
            run("v_mad_u32_u16", 0, false, [0xFFFF, 0xFFFF, 0xFFFF_FFFF], 0),
            0xFFFE_0000
        );
        assert_eq!(
            run("v_mad_u32_u16", 0, true, [0xFFFF, 0xFFFF, 0xFFFF_FFFF], 0),
            0xFFFF_FFFF
        );
        assert_eq!(
            run(
                "v_mad_i32_i16",
                0b11,
                false,
                [0xFFFF_0000, 0x0002_0000, 5],
                0
            ),
            3
        );
        assert_eq!(
            run("v_cvt_pk_u16_u32", 0, false, [0x1_0000, 7, 0], 0),
            0x0007_FFFF
        );
        assert_eq!(
            run("v_cvt_pk_i16_i32", 0, false, [0xFFFF_0000, 0x8000, 0], 0),
            0x7FFF_8000
        );
        assert_eq!(
            run("v_add_nc_i32", 0, false, [0x7FFF_FFFF, 1, 0], 0),
            0x8000_0000
        );
        assert_eq!(
            run("v_add_nc_i32", 0, true, [0x7FFF_FFFF, 1, 0], 0),
            0x7FFF_FFFF
        );
        assert_eq!(
            run("v_sub_nc_i32", 0, true, [0x8000_0000, 1, 0], 0),
            0x8000_0000
        );
    }

    #[test]
    fn the_model_refusals_are_kept() {
        let fields = |abs: u32, neg: u32, omod: u32, clamp: bool| Fields {
            vdst: 0,
            abs,
            sdst: 0,
            clamp,
            sources: [0; 3],
            omod,
            neg,
        };
        let refused = |name: &str, fields: Fields, op_sel: u32| {
            refusal(name, form(name).unwrap(), &fields, op_sel).is_some()
        };
        assert!(refused("v_add_nc_u16", fields(1, 0, 0, false), 0));
        assert!(refused("v_add_nc_u16", fields(0, 2, 0, false), 0));
        assert!(refused("v_add_nc_u16", fields(0, 0, 1, false), 0));
        assert!(!refused("v_add_nc_u16", fields(0, 0, 0, true), 0b1111));
        assert!(refused("v_mul_lo_u16", fields(0, 0, 0, true), 0));
        assert!(refused("v_max3_i16", fields(0, 0, 0, true), 0));
        assert!(refused("v_mad_u32_u16", fields(0, 0, 0, false), 0b0100));
        assert!(!refused("v_mad_u32_u16", fields(0, 0, 0, true), 0b0011));
        assert!(refused("v_cvt_pk_u16_u32", fields(0, 0, 0, false), 1));
        assert!(refused("v_cvt_pk_i16_i32", fields(0, 0, 0, true), 0));
        assert!(refused("v_add_nc_i32", fields(0, 0, 0, false), 0b1000));
        assert!(!refused("v_sub_nc_i32", fields(0, 0, 0, true), 0));
    }
}
