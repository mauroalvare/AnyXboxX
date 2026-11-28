//! Scalar instructions (SOP1, SOP2, SOPC, SOPK, SOPP): port of the `SCALAR` table of
//! `emu/rdna2_exec_ops.py` and of `emu/rdna2_exec_bits.py`. Sources: AMD's « RDNA 2 Instruction Set
//! Architecture », chapter 5. Each function returns `Ok(false)` for a name it does not know, so the
//! caller can name the instruction as untranslated.

use crate::error::Error;
use crate::wave::{EXEC_LO, Wave};

const MASK32: u64 = 0xFFFF_FFFF;
const MASK64: u64 = u64::MAX;
/// « Aucun bit trouvé » : -1 sur 32 bits.
const NO_BIT: u64 = MASK32;

/// The fields of a scalar encoding (the Python `FIELDS` of SOP1, SOP2, SOPC, SOPK and SOPP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fields {
    pub sdst: u32,
    pub ssrc0: u32,
    pub ssrc1: u32,
    pub simm16: u32,
}

/// The scalar fields of a first word. Each family reads only the ones it has.
pub fn fields(word: u32) -> Fields {
    Fields {
        sdst: (word >> 16) & 0x7F,
        ssrc0: word & 0xFF,
        ssrc1: (word >> 8) & 0xFF,
        simm16: word & 0xFFFF,
    }
}

/// `signed(value, bits)` of the Python model: the low `bits` bits read as two's complement.
pub fn sign_extend(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        return value as i64;
    }
    let masked = value & ((1u64 << bits) - 1);
    if masked >> (bits - 1) & 1 == 1 {
        masked as i64 - (1i64 << bits)
    } else {
        masked as i64
    }
}

fn signed32(value: u64) -> i64 {
    sign_extend(value, 32)
}

fn overflow(sum: i64) -> bool {
    !(i64::from(i32::MIN) <= sum && sum <= i64::from(i32::MAX))
}

/// Reads a source with its width, then writes a result: the Python `_sop2`.
fn binary(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    (width, first, second): (usize, usize, usize),
    operation: impl Fn(u64, u64, bool) -> (u64, Option<bool>),
) -> Result<(), Error> {
    let a = wave.read_scalar(f.ssrc0, literal, first)?;
    let b = wave.read_scalar(f.ssrc1, literal, second)?;
    let (result, scc) = operation(a, b, wave.scc);
    wave.write_scalar(f.sdst, result, width)?;
    if let Some(scc) = scc {
        wave.scc = scc;
    }
    Ok(())
}

/// S0 of the signed 64-bit instructions (`s_ashr_i64`, `s_bfe_i64`, `s_flbit_i32_i64`): the Python
/// `signed_literal_s0`. A 32-bit literal there is sign-extended, not zero-extended.
///
/// AMD "RDNA 2" ISA (November 2020), section 6.2.1, "Literal Expansion to 64 bits" (p. 43): "64-bit
/// unsigned integer: zero extended to 64 bits; 64-bit signed integer: sign extended to 64 bits". The
/// unsigned and bitwise 64-bit sources (`s_bfe_u64`, `s_*_b64`) keep zero-extending.
fn read_signed64_s0(wave: &Wave, f: &Fields, literal: Option<u32>) -> Result<u64, Error> {
    let value = wave.read_scalar(f.ssrc0, literal, 2)?;
    Ok(if f.ssrc0 == crate::decode::LITERAL {
        sign_extend(value, 32) as u64
    } else {
        value
    })
}

/// `binary` for the instructions whose S0 is a signed 64-bit integer and whose S1 is 32 bits: a 64-bit
/// result, SCC from the operation.
fn binary_signed64(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    operation: impl Fn(u64, u64) -> (u64, bool),
) -> Result<(), Error> {
    let a = read_signed64_s0(wave, f, literal)?;
    let b = wave.read_scalar(f.ssrc1, literal, 1)?;
    let (result, scc) = operation(a, b);
    wave.write_scalar(f.sdst, result, 2)?;
    wave.scc = scc;
    Ok(())
}

/// The Python `_sop1`: a result of `None` is not written.
fn unary(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    width: usize,
    operation: impl Fn(u64, bool) -> (Option<u64>, Option<bool>),
) -> Result<(), Error> {
    let a = wave.read_scalar(f.ssrc0, literal, width)?;
    let (result, scc) = operation(a, wave.scc);
    if let Some(result) = result {
        wave.write_scalar(f.sdst, result, width)?;
    }
    if let Some(scc) = scc {
        wave.scc = scc;
    }
    Ok(())
}

/// The Python `_logic`: the result masked to the width, SCC = result non-zero.
fn logic(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    width: usize,
    function: fn(u64, u64) -> u64,
) -> Result<(), Error> {
    let mask = if width == 2 { MASK64 } else { MASK32 };
    binary(wave, f, literal, (width, width, width), |a, b, _| {
        let result = function(a, b) & mask;
        (result, Some(result != 0))
    })
}

/// Bits of a `bfe` result: the field of `value` at S1[4:0] (S1[5:0] in 64 bits), width S1[22:16].
fn bfe(value: u64, control: u64, bits: u32, is_signed: bool) -> i64 {
    let offset = (control & u64::from(bits - 1)) as u32;
    let width = ((control >> 16) & 0x7F) as u32;
    if width == 0 {
        return 0;
    }
    if !is_signed {
        let shifted = u128::from(value) >> offset;
        let field = if width >= 64 {
            shifted
        } else {
            shifted & ((1u128 << width) - 1)
        };
        return field as i64;
    }
    let shifted = sign_extend(value, bits) >> offset;
    if width >= bits {
        shifted
    } else {
        sign_extend(shifted as u64, width)
    }
}

/// Whole quad mode: each group of four bits with a bit set becomes 0xF.
fn wqm(value: u64, bits: u32) -> u64 {
    (0..bits)
        .step_by(4)
        .filter(|q| value >> q & 0xF != 0)
        .map(|q| 0xFu64 << q)
        .sum()
}

/// Bit i of the result: one of the bits 4i to 4i+3 of the value is set.
fn quadmask(value: u64, bits: u32) -> u64 {
    (0..bits)
        .step_by(4)
        .filter(|q| value >> q & 0xF != 0)
        .map(|q| 1u64 << (q / 4))
        .sum()
}

/// Each bit i of a 32-bit value copied to bits 2i and 2i+1 of the result.
fn bitreplicate(value: u64) -> u64 {
    (0..32u32)
        .filter(|i| value >> i & 1 == 1)
        .map(|i| 3u64 << (2 * i))
        .sum()
}

/// Leading zeros of a `bits`-bit value, NO_BIT for zero.
fn leading_zeros(value: u64, bits: u32) -> u64 {
    if value == 0 {
        NO_BIT
    } else {
        u64::from(value.leading_zeros() - (64 - bits))
    }
}

/// Position from the top of the first bit that differs from the sign bit, NO_BIT for 0 and -1.
fn leading_signs(value: u64, bits: u32) -> u64 {
    let mask = if bits == 64 {
        MASK64
    } else {
        (1u64 << bits) - 1
    };
    let folded = if value >> (bits - 1) & 1 == 1 {
        !value & mask
    } else {
        value & mask
    };
    leading_zeros(folded, bits)
}

/// Index of the lowest set bit, NO_BIT for zero.
fn first_one(value: u64) -> u64 {
    if value == 0 {
        NO_BIT
    } else {
        u64::from(value.trailing_zeros())
    }
}

fn comparison(condition: &str) -> Option<fn(i128, i128) -> bool> {
    match condition {
        "eq" => Some(|a, b| a == b),
        "lg" => Some(|a, b| a != b),
        "gt" => Some(|a, b| a > b),
        "ge" => Some(|a, b| a >= b),
        "lt" => Some(|a, b| a < b),
        "le" => Some(|a, b| a <= b),
        _ => None,
    }
}

/// Executes one scalar instruction. `Ok(false)` when `name` is not a scalar instruction of this model.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    f: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    if name == "s_version" {
        return Ok(true);
    }
    if crate::movrel::scalar(wave, name, f, literal)? {
        return Ok(true);
    }
    if sop2(wave, name, f, literal)? || sop1(wave, name, f, literal)? {
        return Ok(true);
    }
    if sopc(wave, name, f, literal)? || sopk(wave, name, f)? {
        return Ok(true);
    }
    Ok(sopp_no_operation(name))
}

fn sop2(wave: &mut Wave, name: &str, f: &Fields, lit: Option<u32>) -> Result<bool, Error> {
    let mut done = true;
    match name {
        "s_pack_ll_b32_b16" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            ((b & 0xffff) << 16 | (a & 0xffff), None)
        })?,
        "s_pack_lh_b32_b16" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            ((b & 0xffff_0000) | (a & 0xffff), None)
        })?,
        "s_pack_hh_b32_b16" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            ((b & 0xffff_0000) | (a >> 16), None)
        })?,
        "s_add_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            (a.wrapping_add(b), Some((a + b) >> 32 != 0))
        })?,
        "s_sub_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            (a.wrapping_sub(b), Some(a < b))
        })?,
        "s_addc_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, scc| {
            let total = a + b + u64::from(scc);
            (total, Some(total >> 32 != 0))
        })?,
        "s_subb_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, scc| {
            (
                a.wrapping_sub(b).wrapping_sub(u64::from(scc)),
                Some(a < b + u64::from(scc)),
            )
        })?,
        "s_add_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let sum = signed32(a) + signed32(b);
            (sum as u64, Some(overflow(sum)))
        })?,
        "s_sub_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let difference = signed32(a) - signed32(b);
            (difference as u64, Some(overflow(difference)))
        })?,
        "s_mul_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            ((signed32(a) * signed32(b)) as u64, None)
        })?,
        "s_min_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| (a.min(b), Some(a < b)))?,
        "s_max_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| (a.max(b), Some(a > b)))?,
        "s_min_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            (
                (signed32(a).min(signed32(b))) as u64,
                Some(signed32(a) < signed32(b)),
            )
        })?,
        "s_max_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            (
                (signed32(a).max(signed32(b))) as u64,
                Some(signed32(a) > signed32(b)),
            )
        })?,
        "s_cselect_b32" => binary(wave, f, lit, (1, 1, 1), |a, b, scc| {
            (if scc { a } else { b }, None)
        })?,
        "s_cselect_b64" => binary(wave, f, lit, (2, 2, 2), |a, b, scc| {
            (if scc { a } else { b }, None)
        })?,
        "s_and_b32" => logic(wave, f, lit, 1, |a, b| a & b)?,
        "s_or_b32" => logic(wave, f, lit, 1, |a, b| a | b)?,
        "s_xor_b32" => logic(wave, f, lit, 1, |a, b| a ^ b)?,
        "s_andn2_b32" => logic(wave, f, lit, 1, |a, b| a & !b)?,
        "s_orn2_b32" => logic(wave, f, lit, 1, |a, b| a | !b)?,
        "s_nand_b32" => logic(wave, f, lit, 1, |a, b| !(a & b))?,
        "s_nor_b32" => logic(wave, f, lit, 1, |a, b| !(a | b))?,
        "s_xnor_b32" => logic(wave, f, lit, 1, |a, b| !(a ^ b))?,
        "s_and_b64" => logic(wave, f, lit, 2, |a, b| a & b)?,
        "s_or_b64" => logic(wave, f, lit, 2, |a, b| a | b)?,
        "s_xor_b64" => logic(wave, f, lit, 2, |a, b| a ^ b)?,
        "s_andn2_b64" => logic(wave, f, lit, 2, |a, b| a & !b)?,
        "s_orn2_b64" => logic(wave, f, lit, 2, |a, b| a | !b)?,
        "s_nand_b64" => logic(wave, f, lit, 2, |a, b| !(a & b))?,
        "s_nor_b64" => logic(wave, f, lit, 2, |a, b| !(a | b))?,
        "s_xnor_b64" => logic(wave, f, lit, 2, |a, b| !(a ^ b))?,
        "s_lshl_b32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = (a << (b & 31)) & MASK32;
            (result, Some(result != 0))
        })?,
        "s_lshr_b32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = a >> (b & 31);
            (result, Some(result != 0))
        })?,
        "s_ashr_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = signed32(a) >> (b & 31);
            (result as u64, Some(result != 0))
        })?,
        "s_lshl_b64" => binary(wave, f, lit, (2, 2, 1), |a, b, _| {
            let result = (a << (b & 63)) & MASK64;
            (result, Some(result != 0))
        })?,
        "s_lshr_b64" => binary(wave, f, lit, (2, 2, 1), |a, b, _| {
            let result = a >> (b & 63);
            (result, Some(result != 0))
        })?,
        "s_bfm_b32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            ((((1u64 << (a & 31)) - 1) << (b & 31)), None)
        })?,
        "s_bfe_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = bfe(a, b, 32, false);
            (result as u64, Some(result != 0))
        })?,
        "s_bfe_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = bfe(a, b, 32, true);
            (result as u64, Some(result != 0))
        })?,
        "s_bfe_u64" => binary(wave, f, lit, (2, 2, 1), |a, b, _| {
            let result = bfe(a, b, 64, false);
            (result as u64, Some(result != 0))
        })?,
        "s_bfe_i64" => binary_signed64(wave, f, lit, |a, b| {
            let result = bfe(a, b, 64, true);
            (result as u64, result != 0)
        })?,
        "s_bfm_b64" => binary(wave, f, lit, (2, 1, 1), |a, b, _| {
            let mask = ((1u128 << (a & 63)) - 1) << (b & 63);
            (mask as u64, None)
        })?,
        "s_ashr_i64" => binary_signed64(wave, f, lit, |a, b| {
            let result = sign_extend(a, 64) >> (b & 63);
            (result as u64, result != 0)
        })?,
        "s_absdiff_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            let result = sign_extend(a.wrapping_sub(b), 32).unsigned_abs() & MASK32;
            (result, Some(result != 0))
        })?,
        "s_mul_hi_u32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| ((a * b) >> 32, None))?,
        "s_mul_hi_i32" => binary(wave, f, lit, (1, 1, 1), |a, b, _| {
            (((signed32(a) * signed32(b)) >> 32) as u64, None)
        })?,
        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
            // The digit of « s_lshl<n>_add_u32 », at byte 6.
            let shift = u32::from(name.as_bytes()[6] - b'0');
            binary(wave, f, lit, (1, 1, 1), |a, b, _| {
                let sum = (a << shift) + b;
                (sum, Some(sum >> 32 != 0))
            })?
        }
        _ => done = false,
    }
    Ok(done)
}

fn sop1(wave: &mut Wave, name: &str, f: &Fields, lit: Option<u32>) -> Result<bool, Error> {
    let mut done = true;
    match name {
        "s_mov_b32" => unary(wave, f, lit, 1, |a, _| (Some(a), None))?,
        "s_mov_b64" => unary(wave, f, lit, 2, |a, _| (Some(a), None))?,
        "s_cmov_b32" => unary(wave, f, lit, 1, |a, scc| {
            (if scc { Some(a) } else { None }, None)
        })?,
        "s_cmov_b64" => unary(wave, f, lit, 2, |a, scc| {
            (if scc { Some(a) } else { None }, None)
        })?,
        "s_not_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = !a & MASK32;
            (Some(result), Some(result != 0))
        })?,
        "s_not_b64" => unary(wave, f, lit, 2, |a, _| {
            let result = !a & MASK64;
            (Some(result), Some(result != 0))
        })?,
        "s_brev_b32" => unary(wave, f, lit, 1, |a, _| {
            (Some(u64::from((a as u32).reverse_bits())), None)
        })?,
        "s_brev_b64" => unary(wave, f, lit, 2, |a, _| (Some(a.reverse_bits()), None))?,
        "s_bcnt1_i32_b32" => unary(wave, f, lit, 1, |a, _| {
            (Some(u64::from(a.count_ones())), Some(a != 0))
        })?,
        "s_bcnt0_i32_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = (32 - u64::from((a as u32).count_ones())) & MASK32;
            (Some(result), Some(result != 0))
        })?,
        "s_ff1_i32_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = if a == 0 {
                NO_BIT
            } else {
                u64::from(a.trailing_zeros())
            };
            (Some(result), None)
        })?,
        "s_ff0_i32_b32" => unary(wave, f, lit, 1, |a, _| (Some(first_one(!a & MASK32)), None))?,
        "s_flbit_i32_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = if a == 0 {
                NO_BIT
            } else {
                u64::from((a as u32).leading_zeros())
            };
            (Some(result), None)
        })?,
        "s_flbit_i32" => unary(wave, f, lit, 1, |a, _| (Some(leading_signs(a, 32)), None))?,
        "s_sext_i32_i8" => unary(wave, f, lit, 1, |a, _| {
            (Some(sign_extend(a, 8) as u64), None)
        })?,
        "s_sext_i32_i16" => unary(wave, f, lit, 1, |a, _| {
            (Some(sign_extend(a, 16) as u64), None)
        })?,
        "s_abs_i32" => unary(wave, f, lit, 1, |a, _| {
            (Some(signed32(a).unsigned_abs() & MASK32), Some(a != 0))
        })?,
        "s_wqm_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = wqm(a, 32) & MASK32;
            (Some(result), Some(result != 0))
        })?,
        "s_wqm_b64" => unary(wave, f, lit, 2, |a, _| {
            let result = wqm(a, 64);
            (Some(result), Some(result != 0))
        })?,
        "s_quadmask_b32" => unary(wave, f, lit, 1, |a, _| {
            let result = quadmask(a, 32) & MASK32;
            (Some(result), Some(result != 0))
        })?,
        "s_quadmask_b64" => unary(wave, f, lit, 2, |a, _| {
            let result = quadmask(a, 64);
            (Some(result), Some(result != 0))
        })?,
        "s_bcnt0_i32_b64" => wide_source(
            wave,
            f,
            lit,
            |a| (64 - u64::from(a.count_ones())) & MASK32,
            true,
        )?,
        "s_bcnt1_i32_b64" => {
            wide_source(wave, f, lit, |a| u64::from(a.count_ones()) & MASK32, true)?
        }
        "s_ff0_i32_b64" => wide_source(wave, f, lit, |a| first_one(!a & MASK64) & MASK32, false)?,
        "s_ff1_i32_b64" => wide_source(wave, f, lit, |a| first_one(a) & MASK32, false)?,
        "s_flbit_i32_b64" => wide_source(wave, f, lit, |a| leading_zeros(a, 64) & MASK32, false)?,
        "s_flbit_i32_i64" => {
            let result = leading_signs(read_signed64_s0(wave, f, lit)?, 64) & MASK32;
            wave.write_scalar(f.sdst, result, 1)?;
        }
        "s_bitreplicate_b64_b32" => {
            let a = wave.read_scalar(f.ssrc0, lit, 1)?;
            wave.write_scalar(f.sdst, bitreplicate(a), 2)?;
        }
        "s_bitset0_b32" | "s_bitset1_b32" => {
            bitset(wave, f, lit, 1, u64::from(name.ends_with("1_b32")))?
        }
        "s_bitset0_b64" | "s_bitset1_b64" => {
            bitset(wave, f, lit, 2, u64::from(name.ends_with("1_b64")))?
        }
        _ => done = exec_operation(wave, name, f, lit)?,
    }
    Ok(done)
}

/// A 64-bit source with a 32-bit result, SCC from the result when `scc`.
fn wide_source(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    function: impl Fn(u64) -> u64,
    scc: bool,
) -> Result<(), Error> {
    let result = function(wave.read_scalar(f.ssrc0, literal, 2)?) & MASK32;
    wave.write_scalar(f.sdst, result, 1)?;
    if scc {
        wave.scc = result != 0;
    }
    Ok(())
}

/// s_bitset0/1: D[S0[4:0]] (D[S0[5:0]] in 64 bits) set to `bit`; SCC unchanged.
fn bitset(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    width: usize,
    bit: u64,
) -> Result<(), Error> {
    let bits = 32 * width as u64;
    let index = wave.read_scalar(f.ssrc0, literal, 1)? & (bits - 1);
    let current = wave.read_scalar(f.sdst, literal, width)?;
    let updated = (current & !(1u64 << index)) | (bit << index);
    wave.write_scalar(f.sdst, updated, width)
}

/// s_{op}_saveexec_b32/64 and s_{op}_wrexec_b32/64: EXEC = op(S0, EXEC); D = old or new EXEC.
fn exec_operation(
    wave: &mut Wave,
    name: &str,
    f: &Fields,
    literal: Option<u32>,
) -> Result<bool, Error> {
    let Some(rest) = name.strip_prefix("s_") else {
        return Ok(false);
    };
    let (op, kind) = if let Some(op) = rest.strip_suffix("_b32") {
        (op, 1)
    } else if let Some(op) = rest.strip_suffix("_b64") {
        (op, 2)
    } else {
        return Ok(false);
    };
    let (op, writes_new) = if let Some(op) = op.strip_suffix("_saveexec") {
        (op, false)
    } else if let Some(op) = op.strip_suffix("_wrexec") {
        (op, true)
    } else {
        return Ok(false);
    };
    let combine: fn(u64, u64) -> u64 = match op {
        "and" => |s, e| s & e,
        "or" => |s, e| s | e,
        "xor" => |s, e| s ^ e,
        "andn2" => |s, e| s & !e,
        "orn2" => |s, e| s | !e,
        "nand" => |s, e| !(s & e),
        "nor" => |s, e| !(s | e),
        "xnor" => |s, e| !(s ^ e),
        "andn1" => |s, e| !s & e,
        "orn1" => |s, e| !s | e,
        _ => return Ok(false),
    };
    if writes_new && !matches!(op, "andn1" | "andn2") {
        return Ok(false);
    }
    saveexec(wave, f, literal, kind, combine, writes_new)?;
    Ok(true)
}

fn saveexec(
    wave: &mut Wave,
    f: &Fields,
    literal: Option<u32>,
    width: usize,
    combine: fn(u64, u64) -> u64,
    writes_new: bool,
) -> Result<(), Error> {
    let mask = if width == 2 { MASK64 } else { MASK32 };
    let previous = wave.read_scalar(EXEC_LO, literal, width)?;
    let new = combine(wave.read_scalar(f.ssrc0, literal, width)?, previous) & mask;
    wave.write_scalar(EXEC_LO, new, width)?;
    wave.write_scalar(f.sdst, if writes_new { new } else { previous }, width)?;
    wave.scc = new != 0;
    Ok(())
}

/// s_cmp_<cond>_u32/i32 and the 64-bit forms, s_bitcmp0/1_b32/b64 (SOPC).
fn sopc(wave: &mut Wave, name: &str, f: &Fields, lit: Option<u32>) -> Result<bool, Error> {
    if let Some(rest) = name.strip_prefix("s_cmp_") {
        let (condition, kind) = match rest.split_once('_') {
            Some(pair) => pair,
            None => return Ok(false),
        };
        let Some(test) = comparison(condition) else {
            return Ok(false);
        };
        let (width, signed) = match kind {
            "u32" => (1, false),
            "i32" => (1, true),
            "u64" => (2, false),
            _ => return Ok(false),
        };
        let a = wave.read_scalar(f.ssrc0, lit, width)?;
        let b = wave.read_scalar(f.ssrc1, lit, width)?;
        let (a, b) = if signed {
            (i128::from(signed32(a)), i128::from(signed32(b)))
        } else {
            (i128::from(a), i128::from(b))
        };
        wave.scc = test(a, b);
        return Ok(true);
    }
    let (bits, width) = match name {
        "s_bitcmp0_b32" | "s_bitcmp1_b32" => (31, 1),
        "s_bitcmp0_b64" | "s_bitcmp1_b64" => (63, 2),
        _ => return Ok(false),
    };
    let a = wave.read_scalar(f.ssrc0, lit, width)?;
    let b = wave.read_scalar(f.ssrc1, lit, 1)?;
    let bit = a >> (b & bits) & 1;
    wave.scc = if name.contains("bitcmp0") {
        bit == 0
    } else {
        bit == 1
    };
    Ok(true)
}

/// s_movk, s_cmovk, s_addk, s_mulk and s_cmpk (SOPK). The destination is read as the raw register.
fn sopk(wave: &mut Wave, name: &str, f: &Fields) -> Result<bool, Error> {
    let raw = f.simm16;
    let immediate = sign_extend(u64::from(raw), 16);
    let destination = u64::from(wave.s[f.sdst as usize]);
    match name {
        "s_movk_i32" => wave.write_scalar(f.sdst, immediate as u64, 1)?,
        "s_cmovk_i32" => {
            if wave.scc {
                wave.write_scalar(f.sdst, immediate as u64, 1)?;
            }
        }
        "s_addk_i32" => {
            let sum = signed32(destination) + immediate;
            wave.write_scalar(f.sdst, sum as u64, 1)?;
            wave.scc = overflow(sum);
        }
        "s_mulk_i32" => wave.write_scalar(f.sdst, (signed32(destination) * immediate) as u64, 1)?,
        _ => {
            let Some(rest) = name.strip_prefix("s_cmpk_") else {
                return Ok(false);
            };
            let Some((condition, kind)) = rest.split_once('_') else {
                return Ok(false);
            };
            let Some(test) = comparison(condition) else {
                return Ok(false);
            };
            wave.scc = match kind {
                "u32" => test(i128::from(destination), i128::from(raw)),
                "i32" => test(i128::from(signed32(destination)), i128::from(immediate)),
                _ => return Ok(false),
            };
        }
    }
    Ok(true)
}

/// SOPP instructions with no effect in this model (the Python `NO_OPERATION`). Branches and
/// s_endpgm are handled by the caller.
fn sopp_no_operation(name: &str) -> bool {
    matches!(
        name,
        "s_nop"
            | "s_waitcnt"
            | "s_barrier"
            | "s_sleep"
            | "s_setprio"
            | "s_clause"
            | "s_inst_prefetch"
            | "s_waitcnt_vscnt"
            | "s_waitcnt_vmcnt"
            | "s_waitcnt_expcnt"
            | "s_waitcnt_lgkmcnt"
            | "s_ttracedata"
            | "s_icache_inv"
            | "s_incperflevel"
            | "s_decperflevel"
            | "s_ttracedata_imm"
            | "s_wait_idle"
            | "s_waitcnt_depctr"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_and_overflow_follow_the_model() {
        let mut wave = Wave::new(32).unwrap();
        wave.s[0] = 0xFFFF_FFFF;
        wave.s[1] = 1;
        let f = Fields {
            sdst: 2,
            ssrc0: 0,
            ssrc1: 1,
            simm16: 0,
        };
        assert!(execute(&mut wave, "s_add_u32", &f, None).unwrap());
        assert_eq!((wave.s[2], wave.scc), (0, true));
        wave.s[0] = 0x7FFF_FFFF;
        assert!(execute(&mut wave, "s_add_i32", &f, None).unwrap());
        assert_eq!((wave.s[2], wave.scc), (0x8000_0000, true));
    }

    #[test]
    fn bitfield_extract_reads_offset_and_width_from_the_control_word() {
        assert_eq!(bfe(0xABCD_1234, 4 | (8 << 16), 32, false), 0x23);
        assert_eq!(bfe(0x80, 4 | (4 << 16), 32, true), -8);
        assert_eq!(bfe(0x80, 0, 32, false), 0);
    }

    /// Runs `name` with the literal as S0.
    fn with_literal(wave: &mut Wave, name: &str, sdst: u32, ssrc1: u32, literal: u32) {
        let f = Fields {
            sdst,
            ssrc0: crate::decode::LITERAL,
            ssrc1,
            simm16: 0,
        };
        assert!(execute(wave, name, &f, Some(literal)).unwrap());
    }

    #[test]
    fn a_literal_is_sign_extended_for_a_signed_64_bit_source_only() {
        // ISA section 6.2.1 (p. 43); the cases of tests/test_rdna2_exec_bits.py.
        let mut wave = Wave::new(32).unwrap();
        // s_ashr_i64 by the inline constant 1 (code 129), then by 4 (code 132) on a positive literal.
        with_literal(&mut wave, "s_ashr_i64", 40, 129, 0x8000_0000);
        assert_eq!((wave.s[41], wave.s[40]), (0xFFFF_FFFF, 0xC000_0000));
        assert!(wave.scc);
        with_literal(&mut wave, "s_ashr_i64", 42, 132, 0x7FFF_FFFF);
        assert_eq!((wave.s[43], wave.s[42]), (0, 0x07FF_FFFF));
        // Offset 4, width 40 in s60: the signed extract sees the extended sign, the unsigned one zeros.
        wave.s[60] = 40 << 16 | 4;
        with_literal(&mut wave, "s_bfe_i64", 44, 60, 0x8000_0000);
        assert_eq!((wave.s[45], wave.s[44]), (0xFFFF_FFFF, 0xF800_0000));
        with_literal(&mut wave, "s_bfe_u64", 46, 60, 0x8000_0000);
        assert_eq!((wave.s[47], wave.s[46]), (0, 0x0800_0000));
        // Bit 3 is the first one that differs from the sign; read as bits, 32 leading zeros.
        with_literal(&mut wave, "s_flbit_i32_i64", 51, 0, 0xFFFF_FFF0);
        assert_eq!(wave.s[51], 60);
        with_literal(&mut wave, "s_flbit_i32_b64", 52, 0, 0xFFFF_FFF0);
        assert_eq!(wave.s[52], 32);
        // A register pair is read as it is: no extension of its low half.
        wave.s[0] = 0x8000_0000;
        let pair = Fields {
            sdst: 54,
            ssrc0: 0,
            ssrc1: 129,
            simm16: 0,
        };
        assert!(execute(&mut wave, "s_ashr_i64", &pair, None).unwrap());
        assert_eq!((wave.s[55], wave.s[54]), (0, 0x4000_0000));
    }

    #[test]
    fn unknown_names_are_left_to_the_caller() {
        let mut wave = Wave::new(32).unwrap();
        let f = fields(0);
        assert!(!execute(&mut wave, "s_not_a_real_op", &f, None).unwrap());
    }

    #[test]
    fn saveexec_replaces_exec_and_keeps_the_old_mask_in_sdst() {
        let mut wave = Wave::new(32).unwrap();
        wave.s[0] = 0x0000_00FF;
        let f = Fields {
            sdst: 3,
            ssrc0: 0,
            ssrc1: 0,
            simm16: 0,
        };
        assert!(execute(&mut wave, "s_and_saveexec_b32", &f, None).unwrap());
        assert_eq!(wave.s[3], 0xFFFF_FFFF);
        assert_eq!(wave.s[EXEC_LO as usize], 0xFF);
        assert!(wave.scc);
    }
}
