//! Lane-private LDS proof for console wave64 compute shaders.
//!
//! Console wave64 ray-tracing traversal shaders store a per-thread BVH stack in
//! LDS, either striped across lanes (`(sp << 8) + (tid << 2)` or pointer-stepped
//! `sp += 256 / -= 256`) or in a per-thread slice (`(tid << k) | ((sp & m) << 2)`).
//! Proving that no two lanes of the same 64-lane console wave can access the same
//! 32-bit LDS dword lets `lower.rs` run the shader natively in 32-lane waves on
//! wave32 hardware without `loose_lds` approximation and without `split_lds`
//! per-access group barriers.

use std::collections::HashMap;

use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{NativeProgram, StageIo};
use crate::structure::Structured;

/// Abstract byte address proving disjoint 32-bit LDS dwords across the 64 lanes
/// of a console wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LdsAddr {
    /// Unvisited SSA value.
    Bot,
    /// Exact 32-bit constant.
    Const(u32),
    /// Entry `v0` (`0 <= x < tx`) when flattening a 2D thread group.
    TidX,
    /// Entry `v1` (`0 <= y < ty`) when flattening a 2D thread group.
    TidY,
    /// `tx * v1` when flattening a 2D thread group.
    ScaledTidY,
    /// Value whose low `bits` bits are guaranteed zero (`1..=32`).
    ZeroLow(u8),
    /// Non-negative value `0 <= v <= max` (`max <= 1023`).
    Bounded(u32),
    /// Value satisfying `((v.wrapping_sub(base)) & (M - 1)) == (perm(lane6) << sh) + d`
    /// where `M = 64 << sh` (`sh <= 10`, so `M` divides the 64 KiB LDS space),
    /// `perm(lane6) = (lane6 * odd) & 63` with `odd` odd (a bijection on `0..64`),
    /// `base < M`, and `0 <= d <= span < (1 << sh)`.
    Lane {
        sh: u8,
        odd: u8,
        base: u32,
        span: u32,
    },
    /// Arbitrary / unknown value.
    Top,
}

impl LdsAddr {
    fn join(self, other: Self) -> Self {
        use LdsAddr::*;
        if self == other {
            return self;
        }
        match (self, other) {
            (Bot, x) | (x, Bot) => x,
            (Const(a), Const(b)) => {
                let tz = a.trailing_zeros().min(b.trailing_zeros());
                if tz >= 8 {
                    ZeroLow(tz as u8)
                } else if a <= 1023 && b <= 1023 {
                    Bounded(a.max(b))
                } else if tz > 0 {
                    ZeroLow(tz as u8)
                } else {
                    Top
                }
            }
            (Const(c), ZeroLow(k)) | (ZeroLow(k), Const(c)) => {
                let tz = c.trailing_zeros().min(u32::from(k));
                if tz > 0 { ZeroLow(tz as u8) } else { Top }
            }
            (ZeroLow(a), ZeroLow(b)) => ZeroLow(a.min(b)),
            (Const(c), Bounded(m)) | (Bounded(m), Const(c)) => {
                if c <= 1023 { Bounded(c.max(m)) } else { Top }
            }
            (Bounded(a), Bounded(b)) => Bounded(a.max(b)),
            (
                Lane { sh: s1, odd: o1, base: b1, span: p1 },
                Lane { sh: s2, odd: o2, base: b2, span: p2 },
            ) if s1 == s2 && o1 == o2 => {
                let mask = (64u32 << s1) - 1;
                let slot = 1u32 << s1;
                let d12 = b2.wrapping_sub(b1) & mask;
                let span1 = p1.max(d12.saturating_add(p2));
                let d21 = b1.wrapping_sub(b2) & mask;
                let span2 = p2.max(d21.saturating_add(p1));
                let (base, span) = if span1 <= span2 { (b1, span1) } else { (b2, span2) };
                if span < slot {
                    Lane { sh: s1, odd: o1, base, span }
                } else {
                    Top
                }
            }
            _ => Top,
        }
    }

    fn shl(self, shift: Self, tx: u32) -> Self {
        use LdsAddr::*;
        if self == Bot || shift == Bot {
            return Bot;
        }
        let Const(raw) = shift else { return Top };
        let k = raw & 31;
        if k == 0 {
            return self;
        }
        match self {
            Const(c) => Const(c.wrapping_shl(k)),
            TidY if k <= 5 && tx == (1u32 << k) => ScaledTidY,
            ZeroLow(z) => ZeroLow((u32::from(z) + k).min(32) as u8),
            Bounded(m) => {
                let shifted = u64::from(m) << k;
                if shifted <= 1023 {
                    Bounded(shifted as u32)
                } else {
                    ZeroLow(k as u8)
                }
            }
            Lane { sh, odd, base, span } => {
                let new_sh = u32::from(sh) + k;
                if new_sh <= 10 {
                    let mask = (64u32 << new_sh) - 1;
                    Lane {
                        sh: new_sh as u8,
                        odd,
                        base: base.wrapping_shl(k) & mask,
                        span: span.wrapping_shl(k),
                    }
                } else {
                    ZeroLow(k as u8)
                }
            }
            TidX | TidY | ScaledTidY | Top => ZeroLow(k as u8),
            Bot => Bot,
        }
    }

    fn lshr(self, shift: Self) -> Self {
        use LdsAddr::*;
        if self == Bot || shift == Bot {
            return Bot;
        }
        let Const(raw) = shift else { return Top };
        let k = raw & 31;
        match self {
            Const(c) => Const(c >> k),
            Bounded(m) => Bounded(m >> k),
            _ if k >= 22 => Bounded(u32::MAX >> k),
            _ => Top,
        }
    }

    fn mul(self, other: Self, tx: u32, mask24: bool) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        let (a, b) = match (self, other) {
            (Const(_), _) => (other, self),
            _ => (self, other),
        };
        let Const(c_raw) = b else { return Top };
        let c = if mask24 { c_raw & 0x00FF_FFFF } else { c_raw };
        if c == 0 {
            return Const(0);
        }
        match a {
            Const(x) => {
                let x = if mask24 { x & 0x00FF_FFFF } else { x };
                Const(x.wrapping_mul(c))
            }
            TidY if c == tx && (!mask24 || tx < (1 << 24)) => ScaledTidY,
            Bounded(m) => {
                let prod = u64::from(m) * u64::from(c);
                if prod <= 1023 {
                    Bounded(prod as u32)
                } else if c.trailing_zeros() > 0 {
                    ZeroLow(c.trailing_zeros() as u8)
                } else {
                    Top
                }
            }
            ZeroLow(z) => ZeroLow((u32::from(z) + c.trailing_zeros()).min(32) as u8),
            Lane { sh, odd, base, span } => {
                let tz = c.trailing_zeros();
                let c_odd = c >> tz;
                if c_odd == 1 {
                    return a.shl(Const(tz), tx);
                }
                let new_sh = u32::from(sh) + tz;
                if span == 0 && new_sh <= 10 {
                    let mask = (64u32 << new_sh) - 1;
                    Lane {
                        sh: new_sh as u8,
                        odd: odd.wrapping_mul((c_odd & 63) as u8) & 63,
                        base: base.wrapping_mul(c) & mask,
                        span: 0,
                    }
                } else if tz > 0 {
                    ZeroLow(tz as u8)
                } else {
                    Top
                }
            }
            TidX | TidY | ScaledTidY | Top => {
                if c.trailing_zeros() > 0 {
                    ZeroLow(c.trailing_zeros() as u8)
                } else {
                    Top
                }
            }
            Bot => Bot,
        }
    }

    fn add(self, other: Self) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        match (self, other) {
            (Const(a), Const(b)) => Const(a.wrapping_add(b)),
            (TidX, ScaledTidY) | (ScaledTidY, TidX) => {
                Lane { sh: 0, odd: 1, base: 0, span: 0 }
            }
            (ZeroLow(a), ZeroLow(b)) => ZeroLow(a.min(b)),
            (ZeroLow(a), Const(c)) | (Const(c), ZeroLow(a)) => {
                if c == 0 {
                    ZeroLow(a)
                } else {
                    let tz = c.trailing_zeros().min(u32::from(a));
                    if tz > 0 { ZeroLow(tz as u8) } else { Top }
                }
            }
            (Bounded(a), Const(c)) | (Const(c), Bounded(a)) => {
                if let Some(s) = a.checked_add(c) && s <= 1023 {
                    Bounded(s)
                } else {
                    Top
                }
            }
            (Bounded(a), Bounded(b)) => {
                if let Some(s) = a.checked_add(b) && s <= 1023 {
                    Bounded(s)
                } else {
                    Top
                }
            }
            (Lane { sh, odd, base, span }, Const(c))
            | (Const(c), Lane { sh, odd, base, span }) => {
                let mask = (64u32 << sh) - 1;
                Lane { sh, odd, base: base.wrapping_add(c) & mask, span }
            }
            (Lane { sh, odd, base, span }, ZeroLow(k))
            | (ZeroLow(k), Lane { sh, odd, base, span })
                if u32::from(k) >= 6 + u32::from(sh) =>
            {
                Lane { sh, odd, base, span }
            }
            (Lane { sh, odd, base, span }, Bounded(m))
            | (Bounded(m), Lane { sh, odd, base, span }) => {
                let slot = 1u32 << sh;
                if let Some(s) = span.checked_add(m) && s < slot {
                    Lane { sh, odd, base, span: s }
                } else {
                    Top
                }
            }
            _ => Top,
        }
    }

    fn sub(self, other: Self) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        match (self, other) {
            (Const(a), Const(b)) => Const(a.wrapping_sub(b)),
            (ZeroLow(a), ZeroLow(b)) => ZeroLow(a.min(b)),
            (ZeroLow(a), Const(c)) => {
                if c == 0 {
                    ZeroLow(a)
                } else {
                    let tz = c.trailing_zeros().min(u32::from(a));
                    if tz > 0 { ZeroLow(tz as u8) } else { Top }
                }
            }
            (Lane { sh, odd, base, span }, Const(c)) => {
                let mask = (64u32 << sh) - 1;
                Lane { sh, odd, base: base.wrapping_sub(c) & mask, span }
            }
            (Lane { sh, odd, base, span }, ZeroLow(k))
                if u32::from(k) >= 6 + u32::from(sh) =>
            {
                Lane { sh, odd, base, span }
            }
            _ => Top,
        }
    }

    fn bit_or(self, other: Self, tx: u32) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        match (self, other) {
            (Const(a), Const(b)) => Const(a | b),
            (TidX, ScaledTidY) | (ScaledTidY, TidX) if tx.is_power_of_two() => {
                Lane { sh: 0, odd: 1, base: 0, span: 0 }
            }
            (ZeroLow(a), ZeroLow(b)) => ZeroLow(a.min(b)),
            (ZeroLow(a), Const(c)) | (Const(c), ZeroLow(a)) => {
                if c == 0 {
                    ZeroLow(a)
                } else {
                    let tz = c.trailing_zeros().min(u32::from(a));
                    if tz > 0 { ZeroLow(tz as u8) } else { Top }
                }
            }
            (Bounded(a), Const(c)) | (Const(c), Bounded(a)) => {
                if c <= 1023 && (a | c) <= 1023 {
                    let hi = ((a | c) + 1).next_power_of_two() - 1;
                    if hi <= 1023 { Bounded(hi) } else { Top }
                } else {
                    Top
                }
            }
            (Lane { sh, odd, base, span }, ZeroLow(k))
            | (ZeroLow(k), Lane { sh, odd, base, span })
                if u32::from(k) >= 6 + u32::from(sh) =>
            {
                Lane { sh, odd, base, span }
            }
            (Lane { sh, odd, base, span }, Const(c))
            | (Const(c), Lane { sh, odd, base, span }) => {
                let mask = (64u32 << sh) - 1;
                let slot = 1u32 << sh;
                let low = c & mask;
                if low == 0 {
                    Lane { sh, odd, base, span }
                } else if span == 0 && (base & (slot - 1)) == 0 && low < slot {
                    Lane { sh, odd, base: base | low, span: 0 }
                } else {
                    Top
                }
            }
            (Lane { sh, odd, base, span }, Bounded(m))
            | (Bounded(m), Lane { sh, odd, base, span }) => {
                let slot = 1u32 << sh;
                if span == 0 && (base & (slot - 1)) == 0 && m < slot {
                    Lane { sh, odd, base, span: m }
                } else {
                    Top
                }
            }
            _ => Top,
        }
    }

    fn bit_and(self, other: Self) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        let (a, b) = match (self, other) {
            (Const(_), _) => (other, self),
            _ => (self, other),
        };
        let Const(m) = b else { return Top };
        if m == 0 {
            return Const(0);
        }
        match a {
            Const(c) => Const(c & m),
            Lane { sh, odd, base, span } => {
                let mask = (64u32 << sh) - 1;
                if (m & mask) == mask {
                    Lane { sh, odd, base, span }
                } else if m <= 1023 {
                    Bounded(m)
                } else if m.trailing_zeros() > 0 {
                    ZeroLow(m.trailing_zeros() as u8)
                } else {
                    Top
                }
            }
            Bounded(x) => Bounded(x.min(m)),
            _ if m <= 1023 => Bounded(m),
            _ if m.trailing_zeros() > 0 => ZeroLow(m.trailing_zeros() as u8),
            _ => Top,
        }
    }

    fn bit_xor(self, other: Self) -> Self {
        use LdsAddr::*;
        if self == Bot || other == Bot {
            return Bot;
        }
        match (self, other) {
            (Const(a), Const(b)) => Const(a ^ b),
            (Lane { sh, odd, base, span }, Const(c))
            | (Const(c), Lane { sh, odd, base, span })
                if (c & ((64u32 << sh) - 1)) == 0 =>
            {
                Lane { sh, odd, base, span }
            }
            _ => Top,
        }
    }
}

/// Byte `(offset, width)` pairs accessed by a plain LDS read or write instruction.
/// Returns `None` for atomics, counters, or unrecognized DS instructions.
fn ds_access_layout(name: &str, offset0: u8, offset1: u8) -> Option<Vec<(u32, u32)>> {
    let offset = u32::from(offset0) | (u32::from(offset1) << 8);
    let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
    let sub_bytes = match base {
        "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(1u32),
        "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(2u32),
        _ => None,
    };
    if let Some(bytes) = sub_bytes {
        let valid_suffix = base == name
            || (name.ends_with("_d16") && base.starts_with("ds_read"))
            || name.ends_with("_d16_hi");
        return valid_suffix.then(|| vec![(offset, bytes)]);
    }
    match name {
        "ds_read_b32" | "ds_write_b32" => Some(vec![(offset, 4)]),
        "ds_read_b64" | "ds_write_b64" => Some(vec![(offset, 8)]),
        "ds_read_b96" | "ds_write_b96" => Some(vec![(offset, 12)]),
        "ds_read_b128" | "ds_write_b128" => Some(vec![(offset, 16)]),
        "ds_read2_b32" | "ds_write2_b32" => {
            Some(vec![(u32::from(offset0) * 4, 4), (u32::from(offset1) * 4, 4)])
        }
        "ds_read2_b64" | "ds_write2_b64" => {
            Some(vec![(u32::from(offset0) * 8, 8), (u32::from(offset1) * 8, 8)])
        }
        "ds_read2st64_b32" | "ds_write2st64_b32" => {
            Some(vec![(u32::from(offset0) * 256, 4), (u32::from(offset1) * 256, 4)])
        }
        "ds_read2st64_b64" | "ds_write2st64_b64" => {
            Some(vec![(u32::from(offset0) * 512, 8), (u32::from(offset1) * 512, 8)])
        }
        _ => None,
    }
}

/// Whether every LDS access in a console wave64 compute program is provably
/// lane-private within each 64-lane console wave (no two lanes of the same
/// console wave can touch the same 32-bit LDS dword).
pub(super) fn is_lane_private(p: &Program, st: &Structured, native: &NativeProgram) -> bool {
    let StageIo::Cs(cs) = &native.io else {
        return false;
    };
    if !native.wave64 {
        return false;
    }
    let [tx, ty, tz] = cs.threads;

    // 1. Collect all LDS memory accesses; reject GDS, direct-to-LDS loads, and atomics.
    let mut accesses: Vec<(Value, Vec<(u32, u32)>)> = Vec::new();
    for block in &p.blocks {
        for inst in &block.insts {
            if matches!(
                inst.fields,
                Fields::Buffer { lds: true, .. } | Fields::Flat { lds: true, .. }
            ) {
                return false;
            }
            let name = inst.name();
            if let Fields::Ds { addr, offset0, offset1, gds, .. } = &inst.fields {
                if name.starts_with("ds_swizzle") || name.contains("permute") {
                    continue;
                }
                if *gds {
                    return false;
                }
                let Some(pairs) = ds_access_layout(name, *offset0, *offset1) else {
                    return false;
                };
                let Some(Operand::Vector { reg, .. }) = *addr else {
                    return false;
                };
                let Some(v) = inst.read(Reg::V(reg)) else {
                    return false;
                };
                accesses.push((v, pairs));
            } else if name.starts_with("ds_") {
                return false;
            }
        }
    }
    if accesses.is_empty() {
        return false;
    }

    // 2. Seed program inputs: unknown inputs start at Top; thread IDs give console lane6.
    let mut state: HashMap<Value, LdsAddr> = HashMap::new();
    for &(_, v) in &p.inputs {
        state.insert(v, LdsAddr::Top);
    }
    let tid_vgprs = cs.thread_id_vgprs();
    if tx.is_multiple_of(64) || (ty == 1 && tz == 1) {
        if tid_vgprs < 1 {
            return false;
        }
        for &(reg, v) in &p.inputs {
            if reg == Reg::V(0) {
                state.insert(v, LdsAddr::Lane { sh: 0, odd: 1, base: 0, span: 0 });
            }
        }
    } else if tx.saturating_mul(ty).is_multiple_of(64) || tz == 1 {
        if tid_vgprs < 2 {
            return false;
        }
        for &(reg, v) in &p.inputs {
            match reg {
                Reg::V(0) => {
                    state.insert(v, LdsAddr::TidX);
                }
                Reg::V(1) => {
                    state.insert(v, LdsAddr::TidY);
                }
                _ => {}
            }
        }
    } else {
        return false;
    }

    let op_val = |inst: &Inst, op: &Operand, state: &HashMap<Value, LdsAddr>| -> LdsAddr {
        match *op {
            Operand::Const(c @ 128..=192) => LdsAddr::Const(u32::from(c) - 128),
            Operand::Const(c @ 193..=208) => LdsAddr::Const((-(i32::from(c) - 192)) as u32),
            Operand::Literal(lit) => LdsAddr::Const(lit),
            Operand::Vector { reg, .. } => inst
                .read(Reg::V(reg))
                .and_then(|v| state.get(&v).copied())
                .unwrap_or(LdsAddr::Bot),
            Operand::Scalar { code, .. } if code == crate::gfx10::NULL => LdsAddr::Const(0),
            Operand::Scalar { code, .. } => inst
                .read(Reg::S(code))
                .and_then(|v| state.get(&v).copied())
                .unwrap_or(LdsAddr::Bot),
            _ => LdsAddr::Top,
        }
    };

    let eval_inst = |inst: &Inst, state: &HashMap<Value, LdsAddr>| -> Option<(Reg, LdsAddr)> {
        let name = inst.name();
        match &inst.fields {
            Fields::Valu(v) => {
                if v.dpp.is_some()
                    || v.sdwa.is_some()
                    || v.opsel != 0
                    || v.clamp
                    || v.omod != 0
                    || v.abs != 0
                    || v.neg != 0
                {
                    return None;
                }
                let Operand::Vector { reg, dwords: 1 } = v.dst? else {
                    return None;
                };
                let s = |k: usize| v.src.get(k).map(|op| op_val(inst, op, state));
                let out = match name {
                    "v_mov_b32" => s(0)?,
                    "v_cndmask_b32" => s(0)?.join(s(1)?),
                    "v_lshlrev_b32" => s(1)?.shl(s(0)?, tx),
                    "v_lshrrev_b32" => s(1)?.lshr(s(0)?),
                    "v_add_nc_u32" | "v_add_co_u32" => s(0)?.add(s(1)?),
                    "v_sub_nc_u32" | "v_sub_co_u32" => s(0)?.sub(s(1)?),
                    "v_subrev_nc_u32" | "v_subrev_co_u32" => s(1)?.sub(s(0)?),
                    "v_or_b32" => s(0)?.bit_or(s(1)?, tx),
                    "v_and_b32" => s(0)?.bit_and(s(1)?),
                    "v_xor_b32" => s(0)?.bit_xor(s(1)?),
                    "v_mul_lo_u32" => s(0)?.mul(s(1)?, tx, false),
                    "v_mul_u32_u24" => s(0)?.mul(s(1)?, tx, true),
                    "v_lshl_add_u32" => s(0)?.shl(s(1)?, tx).add(s(2)?),
                    "v_lshl_or_b32" => s(0)?.shl(s(1)?, tx).bit_or(s(2)?, tx),
                    "v_add_lshl_u32" => s(0)?.add(s(1)?).shl(s(2)?, tx),
                    "v_add3_u32" => s(0)?.add(s(1)?).add(s(2)?),
                    "v_or3_b32" => s(0)?.bit_or(s(1)?, tx).bit_or(s(2)?, tx),
                    "v_and_or_b32" => s(0)?.bit_and(s(1)?).bit_or(s(2)?, tx),
                    "v_mad_u32_u24" => s(0)?.mul(s(1)?, tx, true).add(s(2)?),
                    "v_bfe_u32" => match (s(0)?, s(1)?, s(2)?) {
                        (LdsAddr::Bot, _, _) | (_, LdsAddr::Bot, _) | (_, _, LdsAddr::Bot) => {
                            LdsAddr::Bot
                        }
                        (_, _, LdsAddr::Const(0)) => LdsAddr::Const(0),
                        (LdsAddr::Const(c), LdsAddr::Const(off), LdsAddr::Const(w)) => {
                            let w = w & 31;
                            let shifted = c.wrapping_shr(off & 31);
                            LdsAddr::Const(if w == 0 { 0 } else { shifted & ((1u32 << w) - 1) })
                        }
                        (
                            LdsAddr::Lane { sh: 0, odd, base, span: 0 },
                            LdsAddr::Const(0),
                            LdsAddr::Const(w),
                        ) if (w & 31) >= 6 => LdsAddr::Lane { sh: 0, odd, base, span: 0 },
                        (_, LdsAddr::Const(_), LdsAddr::Const(w))
                            if (w & 31) > 0 && (w & 31) <= 10 =>
                        {
                            LdsAddr::Bounded((1u32 << (w & 31)) - 1)
                        }
                        _ => LdsAddr::Top,
                    },
                    _ => return None,
                };
                Some((Reg::V(reg), out))
            }
            Fields::Sop { dst, src, imm } => {
                let Operand::Scalar { code, dwords: 1 } = (*dst)? else {
                    return None;
                };
                let s = |k: usize| src.get(k).map(|op| op_val(inst, op, state));
                let out = match name {
                    "s_mov_b32" => s(0)?,
                    "s_movk_i32" => LdsAddr::Const(i32::from((*imm)? as i16) as u32),
                    "s_cselect_b32" => s(0)?.join(s(1)?),
                    "s_lshl_b32" => s(0)?.shl(s(1)?, tx),
                    "s_lshr_b32" => s(0)?.lshr(s(1)?),
                    "s_add_u32" | "s_add_i32" => s(0)?.add(s(1)?),
                    "s_addk_i32" => {
                        let k = LdsAddr::Const(i32::from((*imm)? as i16) as u32);
                        s(0)?.add(k)
                    }
                    "s_sub_u32" | "s_sub_i32" => s(0)?.sub(s(1)?),
                    "s_or_b32" => s(0)?.bit_or(s(1)?, tx),
                    "s_and_b32" => s(0)?.bit_and(s(1)?),
                    "s_xor_b32" => s(0)?.bit_xor(s(1)?),
                    "s_mul_i32" => s(0)?.mul(s(1)?, tx, false),
                    "s_mulk_i32" => {
                        let k = LdsAddr::Const(i32::from((*imm)? as i16) as u32);
                        s(0)?.mul(k, tx, false)
                    }
                    _ => return None,
                };
                Some((Reg::S(code), out))
            }
            _ => None,
        }
    };

    // 3. Fixed-point dataflow in RPO order with widening at loop PHIs.
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut converged = false;
    for iter in 0..64u32 {
        let mut changed = false;
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                let mut acc = LdsAddr::Bot;
                for &arg in &phi.args {
                    let a = state.get(&arg).copied().unwrap_or(LdsAddr::Bot);
                    acc = acc.join(a);
                }
                let prev = state.get(&phi.value).copied().unwrap_or(LdsAddr::Bot);
                let next = if iter >= 8 && acc != prev {
                    match (prev, acc) {
                        (LdsAddr::Bounded(_), LdsAddr::Bounded(_))
                        | (LdsAddr::Lane { .. }, LdsAddr::Lane { .. }) => LdsAddr::Top,
                        _ => prev.join(acc),
                    }
                } else {
                    prev.join(acc)
                };
                if next != prev {
                    state.insert(phi.value, next);
                    changed = true;
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let computed = eval_inst(inst, &state);
                let guarded = st
                    .guards
                    .get(b)
                    .and_then(|g| g.get(i))
                    .copied()
                    .flatten()
                    .is_some();
                for &(reg, val) in &inst.writes {
                    let rhs = match computed {
                        Some((r, a)) if r == reg => a,
                        _ => LdsAddr::Top,
                    };
                    let combined = if guarded && matches!(reg, Reg::V(_)) {
                        let old = inst
                            .read(reg)
                            .and_then(|v| state.get(&v).copied())
                            .unwrap_or(LdsAddr::Top);
                        rhs.join(old)
                    } else {
                        rhs
                    };
                    let prev = state.get(&val).copied().unwrap_or(LdsAddr::Bot);
                    let next = prev.join(combined);
                    if next != prev {
                        state.insert(val, next);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return false;
    }

    // 4. Verify all DS accesses share one `(sh, odd)` (`2 <= sh <= 10`) and their
    //    enclosing 4-byte dwords fit inside a single `(1 << sh)`-byte window per lane.
    let mut global_key: Option<(u8, u8)> = None;
    let mut global_arc: Option<(u32, u32)> = None;
    for (addr_val, pairs) in accesses {
        let Some(LdsAddr::Lane { sh, odd, base, span }) = state.get(&addr_val).copied() else {
            return false;
        };
        if !(2..=10).contains(&sh) {
            return false;
        }
        match global_key {
            None => global_key = Some((sh, odd)),
            Some(k) if k == (sh, odd) => {}
            Some(_) => return false,
        }
        let mask = (64u32 << sh) - 1;
        let slot = 1u32 << sh;
        for (byte_off, byte_width) in pairs {
            let start = base.wrapping_add(byte_off) & mask;
            let lo_dw = start & !3;
            let end_byte = start.saturating_add(span).saturating_add(byte_width);
            let hi_dw = (end_byte.saturating_sub(1) | 3).saturating_add(1);
            let len = hi_dw - lo_dw;
            if len > slot {
                return false;
            }
            global_arc = Some(match global_arc {
                None => (lo_dw, len),
                Some((g_base, g_len)) => {
                    let d12 = lo_dw.wrapping_sub(g_base) & mask;
                    let len1 = g_len.max(d12.saturating_add(len));
                    let d21 = g_base.wrapping_sub(lo_dw) & mask;
                    let len2 = len.max(d21.saturating_add(g_len));
                    let (nb, nl) = if len1 <= len2 { (g_base, len1) } else { (lo_dw, len2) };
                    if nl > slot {
                        return false;
                    }
                    (nb, nl)
                }
            });
        }
    }
    true
}
