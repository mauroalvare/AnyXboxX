//! The state of a wave: scalar registers, vector registers, SCC, the program counter (port of the
//! `Wave` class of `emu/rdna2_exec.py`).
//!
//! `v[n][lane]` are the vector registers; `s[n]` the 32-bit scalar registers. In wave64, EXEC and
//! VCC take two SGPR each (a lane mask is a register pair), as in AMD's documentation.

use std::collections::BTreeMap;

use crate::error::Error;
use crate::float;
use crate::memory::Memory;

/// SGPRs: s0 to s105, then VCC, TTMP, M0, NULL, EXEC.
pub const SCALAR_REGISTERS: usize = 128;
pub const VCC_LO: u32 = 106;
pub const M0: u32 = 124;
pub const NULL: u32 = 125;
pub const EXEC_LO: u32 = 126;
/// First vector register as a source operand (v0).
pub const VGPR_FIRST: u32 = 256;
/// Largest number of instructions one run may execute before it is called a loop.
pub const MAX_STEPS: u64 = 1_000_000;
const VECTOR_REGISTERS: usize = 256;

/// The state of one wave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave {
    pub lanes: usize,
    pub mask_width: usize,
    pub s: Vec<u32>,
    pub v: Vec<Vec<u32>>,
    pub scc: bool,
    pub pc: usize,
    pub steps: u64,
    pub memory: Memory,
    /// The local data share of the wave (`ds::LDS_SIZE` bytes).
    pub lds: Vec<u8>,
    /// The EXP exports: target, then lane, then the four values (`wave.exports` in the Python).
    pub exports: BTreeMap<u32, BTreeMap<usize, [u32; 4]>>,
    /// The primitive VINTRP reads the parameters of (`wave.primitive` in the Python).
    pub primitive: u64,
    /// Per-lane primitives, when given (`wave.primitives`); a lane it lacks takes `primitive`.
    pub primitives: Option<BTreeMap<usize, u64>>,
}

impl Wave {
    /// A wave of 32 or 64 lanes with EXEC set to every lane; v0 holds the lane numbers.
    pub fn new(lanes: usize) -> Result<Wave, Error> {
        if lanes != 32 && lanes != 64 {
            return Err(Error::Operand(format!(
                "taille de wave non prise en charge : {lanes} (32 ou 64)"
            )));
        }
        let mut wave = Wave {
            lanes,
            mask_width: lanes / 32,
            s: vec![0; SCALAR_REGISTERS],
            v: vec![vec![0; lanes]; VECTOR_REGISTERS],
            scc: false,
            pc: 0,
            steps: 0,
            memory: Memory::default(),
            lds: vec![0; crate::ds::LDS_SIZE],
            exports: BTreeMap::new(),
            primitive: 0,
            primitives: None,
        };
        let all = if lanes == 64 {
            u64::MAX
        } else {
            (1u64 << lanes) - 1
        };
        wave.write_mask(EXEC_LO, all)?;
        wave.v[0] = (0..lanes as u32).collect();
        Ok(wave)
    }

    pub fn exec(&self) -> Result<u64, Error> {
        self.read_mask(EXEC_LO)
    }

    pub fn vcc(&self) -> Result<u64, Error> {
        self.read_mask(VCC_LO)
    }

    /// The lanes whose EXEC bit is set, in order.
    pub fn active(&self) -> Result<Vec<usize>, Error> {
        let mask = self.exec()?;
        Ok((0..self.lanes)
            .filter(|lane| mask >> lane & 1 == 1)
            .collect())
    }

    pub fn read_mask(&self, code: u32) -> Result<u64, Error> {
        self.read_scalar(code, None, self.mask_width)
    }

    pub fn write_mask(&mut self, code: u32, mask: u64) -> Result<(), Error> {
        self.write_scalar(code, mask, self.mask_width)
    }

    /// A scalar source operand, `width` registers wide (1 or 2), as the Python `read_scalar`.
    pub fn read_scalar(&self, code: u32, literal: Option<u32>, width: usize) -> Result<u64, Error> {
        if code == crate::decode::LITERAL {
            return literal
                .map(u64::from)
                .ok_or_else(|| Error::Operand("constante littérale absente".into()));
        }
        if code == NULL {
            return Ok(0);
        }
        if (code as usize) < SCALAR_REGISTERS {
            let index = code as usize;
            let low = u64::from(self.s[index]);
            let high = if width == 2 && index + 1 < SCALAR_REGISTERS {
                u64::from(self.s[index + 1]) << 32
            } else {
                0
            };
            return Ok(low | high);
        }
        match code {
            128..=192 => Ok(u64::from(code - 128)),
            193..=208 => {
                let value = (192i64 - i64::from(code)) as u64;
                Ok(if width == 1 {
                    value & 0xFFFF_FFFF
                } else {
                    value
                })
            }
            _ => (if width == 2 {
                float::inline_double(code)
            } else {
                float::inline_constant(code)
            })
            .ok_or_else(|| {
                Error::Untranslated(format!("opérande scalaire {code} non pris en charge"))
            }),
        }
    }

    /// Writes a scalar destination of `width` registers; NULL discards, others are masked to 32 bits.
    pub fn write_scalar(&mut self, code: u32, value: u64, width: usize) -> Result<(), Error> {
        if code == NULL {
            return Ok(());
        }
        let index = code as usize;
        if index
            .checked_add(width)
            .is_none_or(|end| end > SCALAR_REGISTERS)
        {
            return Err(Error::Untranslated(format!(
                "écriture de {} bits au-delà de s127 (s{code})",
                32 * width
            )));
        }
        self.s[index] = value as u32;
        if width == 2 {
            self.s[index + 1] = (value >> 32) as u32;
        }
        Ok(())
    }

    /// The index of the first of `count` consecutive VGPRs starting at `first`. A group that runs past
    /// v255 (a 64-bit pair at v255, the four registers of a 128-bit value at v253) is an operand the
    /// model refuses: the Python raises IndexError there, and indexing `v` directly would panic.
    pub fn vgpr_group(&self, first: usize, count: usize) -> Result<usize, Error> {
        if first + count > self.v.len() {
            return Err(Error::Operand(format!(
                "groupe de {count} registres vectoriels hors plage : v{first}"
            )));
        }
        Ok(first)
    }

    /// A source of a vector instruction for one lane: a vector register, or a scalar operand.
    pub fn read_lane(&self, code: u32, lane: usize, literal: Option<u32>) -> Result<u32, Error> {
        if code >= VGPR_FIRST {
            return Ok(self.v[(code - VGPR_FIRST) as usize][lane]);
        }
        Ok(self.read_scalar(code, literal, 1)? as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_register_group_must_end_at_v255() {
        let wave = Wave::new(32).unwrap();
        assert_eq!(wave.vgpr_group(255, 1).unwrap(), 255);
        assert_eq!(wave.vgpr_group(254, 2).unwrap(), 254);
        assert_eq!(wave.vgpr_group(252, 4).unwrap(), 252);
        assert!(matches!(wave.vgpr_group(255, 2), Err(Error::Operand(_))));
        assert!(matches!(wave.vgpr_group(253, 4), Err(Error::Operand(_))));
    }

    #[test]
    fn mask_write_past_s127_is_refused_without_mutation() {
        let mut wave = Wave::new(64).unwrap();
        wave.s[127] = 0x1234_5678;
        assert!(matches!(
            wave.write_mask(127, 0),
            Err(Error::Untranslated(_))
        ));
        assert_eq!(wave.s[127], 0x1234_5678);
    }

    #[test]
    fn exec_starts_with_every_lane_and_v0_holds_the_lane_numbers() {
        let wave = Wave::new(32).unwrap();
        assert_eq!(wave.exec().unwrap(), 0xFFFF_FFFF);
        assert_eq!(wave.v[0][31], 31);
        assert_eq!(Wave::new(64).unwrap().exec().unwrap(), u64::MAX);
    }

    #[test]
    fn inline_and_literal_operands_read_as_the_python_model_does() {
        let mut wave = Wave::new(32).unwrap();
        assert_eq!(wave.read_scalar(128 + 5, None, 1).unwrap(), 5);
        assert_eq!(wave.read_scalar(193, None, 1).unwrap(), 0xFFFF_FFFF);
        assert_eq!(wave.read_scalar(193, None, 2).unwrap(), u64::MAX);
        assert_eq!(wave.read_scalar(255, Some(7), 1).unwrap(), 7);
        wave.write_scalar(3, 0x1_0000_0002, 1).unwrap();
        assert_eq!(wave.s[3], 2);
        assert_eq!(wave.read_scalar(125, None, 1).unwrap(), 0);
    }

    #[test]
    fn unknown_operands_are_untranslated_and_named() {
        let wave = Wave::new(32).unwrap();
        assert!(matches!(
            wave.read_scalar(251, None, 1),
            Err(Error::Untranslated(_))
        ));
    }
}
