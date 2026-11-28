//! The image as it lies in memory: headers and sections at their RVAs, base relocations, and the
//! protection of each page (`loader._map` and `_protection`).

use crate::headers::Headers;
use crate::{LoadError, MAX_IMAGE, PAGE, align};
use victory_pe::layout::dir;

/// `IMAGE_DIRECTORY_ENTRY_BASERELOC`.
pub const DIR_RELOC: usize = 5;
/// File ranges for the certificate and bound-import tables.
const DIR_SECURITY: usize = 4;
const DIR_BOUND_IMPORT: usize = 11;
const DIR_EXPORT: usize = 0;

const SCN_EXEC: u32 = 0x2000_0000;
const SCN_READ: u32 = 0x4000_0000;
const SCN_WRITE: u32 = 0x8000_0000;

/// Page protection bits, the same values as Unicorn's `UC_PROT_*`.
pub const PROT_READ: u8 = 1;
pub const PROT_WRITE: u8 = 2;
pub const PROT_EXEC: u8 = 4;

const REL_ABSOLUTE: u16 = 0;
const REL_HIGHLOW: u16 = 3;
const REL_DIR64: u16 = 10;
const RELOC_BLOCK_HEADER: usize = 8;

/// File and image ranges checked before allocating either the image or its page table.
pub(crate) struct ImageLayout<'a> {
    data: &'a [u8],
    headers: &'a Headers,
    size: usize,
    header_len: usize,
}

impl<'a> ImageLayout<'a> {
    pub(crate) fn check(data: &'a [u8], headers: &'a Headers) -> Result<Self, LoadError> {
        let image_size = u64::from(headers.size_of_image);
        let file_len = data.len() as u64;
        let declared_headers = u64::from(headers.size_of_headers);
        let header_len = if declared_headers > file_len {
            let section_start = headers
                .sections
                .first()
                .map_or(image_size, |section| u64::from(section.virtual_address));
            file_len.min(section_start)
        } else {
            declared_headers
        };
        if image_size > MAX_IMAGE {
            return Err(LoadError::TooLarge);
        }
        if image_size == 0
            || header_len > image_size
            || header_len < headers.section_table_end as u64
        {
            return Err(LoadError::BadHeaders);
        }
        let section_alignment = u64::from(headers.section_alignment);
        let file_alignment = u64::from(headers.file_alignment);
        if !section_alignment.is_power_of_two()
            || !file_alignment.is_power_of_two()
            || file_alignment > section_alignment
            || (section_alignment < PAGE && section_alignment != file_alignment)
            || !image_size.is_multiple_of(section_alignment)
        {
            return Err(LoadError::Alignment);
        }

        let mut last_end = header_len;
        for section in &headers.sections {
            // Widen the input fields before adding. Either sum can exceed a 32-bit PE RVA.
            let start = u64::from(section.virtual_address);
            let length = u64::from(section.mapped_size());
            let end = start + length.div_ceil(section_alignment) * section_alignment;
            if !start.is_multiple_of(section_alignment) || start < last_end || end > image_size {
                return Err(LoadError::Section(start));
            }
            let raw = u64::from(section.raw_offset);
            let raw_size = u64::from(section.raw_size);
            // Use the same declared file offset as the import and export parser.
            if raw_size != 0
                && (raw < headers.section_table_end as u64 || raw + raw_size > file_len)
            {
                return Err(LoadError::RawSection(start));
            }
            last_end = end;
        }
        for (index, bound) in [
            (DIR_EXPORT, image_size),
            (dir::IMPORT, image_size),
            (DIR_RELOC, image_size),
            (crate::tls::DIR_TLS, image_size),
            (crate::DIR_IMPORT_DELAY, image_size),
            (DIR_SECURITY, file_len),
            (DIR_BOUND_IMPORT, file_len),
        ] {
            let (address, length) = headers.directory(index);
            if address != 0 && u64::from(address) + u64::from(length) > bound {
                return Err(LoadError::DirectoryRange(index));
            }
        }
        Ok(Self {
            data,
            headers,
            size: align(image_size) as usize,
            header_len: header_len as usize,
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn map(&self) -> Result<Vec<u8>, LoadError> {
        let mut image = allocate_filled(self.size, 0)?;
        let header_len = self.header_len;
        image[..header_len].copy_from_slice(&self.data[..header_len]);
        for section in self
            .headers
            .sections
            .iter()
            .filter(|section| section.raw_size != 0)
        {
            let src = section.raw_offset as usize;
            let dst = section.virtual_address as usize;
            let len = section.copied_size() as usize;
            image[dst..dst + len].copy_from_slice(&self.data[src..src + len]);
        }
        Ok(image)
    }

    pub(crate) fn page_protections(&self) -> Result<Vec<u8>, LoadError> {
        for section in &self.headers.sections {
            let rva = u64::from(section.virtual_address);
            if !rva.is_multiple_of(PAGE) {
                return Err(LoadError::Section(rva));
            }
        }
        let mut pages = allocate_filled(self.size / PAGE as usize, PROT_READ)?;
        for section in &self.headers.sections {
            let first = section.virtual_address as usize / PAGE as usize;
            let length = align(u64::from(section.mapped_size()));
            let count = (length / PAGE) as usize;
            pages[first..first + count].fill(protection(section.characteristics));
        }
        Ok(pages)
    }

    /// An RVA span directly backed by file bytes, when wholly in one header or section range.
    fn file_bytes(&self, at: usize, len: usize) -> Option<&'a [u8]> {
        let header_len = self.header_len;
        if at < header_len {
            return (at + len <= header_len).then(|| &self.data[at..at + len]);
        }
        let sections = &self.headers.sections;
        let index = sections.partition_point(|section| section.virtual_address as usize <= at);
        let section = sections.get(index.checked_sub(1)?)?;
        let offset = at - section.virtual_address as usize;
        if offset + len > section.copied_size() as usize {
            return None;
        }
        let raw = section.raw_offset as usize + offset;
        self.data.get(raw..raw + len)
    }

    pub(crate) fn relocations(&self) -> Result<Relocations<'_, 'a>, LoadError> {
        let (rva, size) = self.headers.directory(DIR_RELOC);
        let (start, length) = (rva as usize, if rva == 0 { 0 } else { size as usize });
        let mut table = Relocations {
            layout: self,
            start,
            length,
            file_bytes: self.file_bytes(start, length),
            has_entries: false,
        };
        let image_size = u64::from(self.headers.size_of_image);
        table.has_entries = table.visit(|rva, kind| {
            let width = relocation_width(kind)?;
            if width != 0 && rva + width > image_size {
                return Err(LoadError::Relocation(format!(
                    "field at {rva:#x} outside the image"
                )));
            }
            Ok(())
        })?;
        Ok(table)
    }
}

fn allocate_filled(bytes: usize, value: u8) -> Result<Vec<u8>, LoadError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(bytes)
        .map_err(|_| LoadError::Allocation(bytes))?;
    result.resize(bytes, value);
    Ok(result)
}

pub(crate) struct Relocations<'layout, 'file> {
    layout: &'layout ImageLayout<'file>,
    start: usize,
    length: usize,
    file_bytes: Option<&'file [u8]>,
    has_entries: bool,
}

impl Relocations<'_, '_> {
    pub(crate) fn has_entries(&self) -> bool {
        self.has_entries
    }

    pub(crate) fn apply(&self, image: &mut [u8], delta: u64) -> Result<(), LoadError> {
        self.visit(|rva, kind| relocate_field(image, rva, kind, delta))?;
        Ok(())
    }

    fn read<const N: usize>(&self, offset: usize) -> [u8; N] {
        if let Some(table) = self.file_bytes {
            return table[offset..offset + N]
                .try_into()
                .expect("checked relocation entry range");
        }
        // A directory can span mapped ranges. Read the same header, section and zero-fill bytes
        // that would be present in the mapped image, without reserving it first.
        std::array::from_fn(|i| {
            self.layout
                .file_bytes(self.start + offset + i, 1)
                .map_or(0, |bytes| bytes[0])
        })
    }

    fn visit(
        &self,
        mut entry: impl FnMut(u64, u16) -> Result<(), LoadError>,
    ) -> Result<bool, LoadError> {
        let mut at = 0;
        let mut found = false;
        while at < self.length {
            let rva = self.start + at;
            if self.length - at < RELOC_BLOCK_HEADER {
                return Err(LoadError::Relocation(format!(
                    "truncated block at {rva:#x}"
                )));
            }
            let page = u64::from(u32::from_le_bytes(self.read(at)));
            let block = u32::from_le_bytes(self.read(at + 4)) as usize;
            // A zero-length block ends the relocation table, matching pefile.
            if block == 0 {
                break;
            }
            if block < RELOC_BLOCK_HEADER || block > self.length - at {
                return Err(LoadError::Relocation(format!(
                    "malformed block at {rva:#x}"
                )));
            }
            for offset in (at + RELOC_BLOCK_HEADER..at + block - 1).step_by(2) {
                let value = u16::from_le_bytes(self.read(offset));
                entry(page + u64::from(value & 0xFFF), value >> 12)?;
                found = true;
            }
            at += block;
        }
        Ok(found)
    }
}

fn relocation_width(kind: u16) -> Result<u64, LoadError> {
    match kind {
        REL_ABSOLUTE => Ok(0),
        REL_DIR64 => Ok(8),
        REL_HIGHLOW => Ok(4),
        other => Err(LoadError::Relocation(format!("unsupported type {other}"))),
    }
}

fn relocate_field(image: &mut [u8], rva: u64, kind: u16, delta: u64) -> Result<(), LoadError> {
    let width = relocation_width(kind)?;
    if width == 0 {
        return Ok(());
    }
    let width = width as usize;
    let offset = rva as usize;
    let field = offset
        .checked_add(width)
        .and_then(|end| image.get_mut(offset..end))
        .ok_or_else(|| LoadError::Relocation(format!("field at {rva:#x} outside the image")))?;
    if kind == REL_DIR64 {
        let value = u64::from_le_bytes(field.try_into().expect("8-byte field"));
        field.copy_from_slice(&value.wrapping_add(delta).to_le_bytes());
    } else {
        let value = u32::from_le_bytes(field.try_into().expect("4-byte field"));
        field.copy_from_slice(&value.wrapping_add(delta as u32).to_le_bytes());
    }
    Ok(())
}

fn protection(flags: u32) -> u8 {
    let readable = flags & (SCN_READ | SCN_WRITE | SCN_EXEC) != 0;
    (if readable { PROT_READ } else { 0 })
        | (if flags & SCN_WRITE != 0 {
            PROT_WRITE
        } else {
            0
        })
        | (if flags & SCN_EXEC != 0 { PROT_EXEC } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_writable_or_executable_section_stays_readable() {
        assert_eq!(protection(SCN_WRITE), PROT_READ | PROT_WRITE);
        assert_eq!(protection(SCN_EXEC), PROT_READ | PROT_EXEC);
        assert_eq!(protection(0), 0);
    }

    #[test]
    fn dir64_and_highlow_fields_move_by_the_delta() {
        let mut image = vec![0u8; 16];
        image[..8].copy_from_slice(&0x1_4000_1000u64.to_le_bytes());
        image[8..12].copy_from_slice(&0x1000u32.to_le_bytes());
        relocate_field(&mut image, 0, REL_DIR64, 0x10_0000).unwrap();
        relocate_field(&mut image, 8, REL_HIGHLOW, 0x10_0000).unwrap();
        relocate_field(&mut image, 4, REL_ABSOLUTE, 0x10_0000).unwrap();
        assert_eq!(&image[..8], &0x1_4010_1000u64.to_le_bytes());
        assert_eq!(&image[8..12], &0x10_1000u32.to_le_bytes());
    }

    #[test]
    fn unknown_relocation_types_are_refused() {
        let mut image = vec![0u8; 16];
        assert!(matches!(
            relocate_field(&mut image, 0, 4, 1),
            Err(LoadError::Relocation(_))
        ));
        assert!(relocate_field(&mut image, 12, REL_DIR64, 1).is_err());
    }
}
