//! The header fields AnyXboxX checks before loading an image, read straight from the file.
//!
//! `victory_pe::ImageInfo` parses the tables (imports, exports, TLS callbacks); these few fields
//! are read here because the checks of `loader.py` need them before any table is trusted, and
//! some (`SizeOfHeaders`, `Magic`, raw section characteristics) are not part of `ImageInfo`.

use victory_pe::layout::{self, read_u32, read_u64};

use crate::LoadError;

const PE_SIGNATURE: &[u8; 4] = b"PE\0\0";
const MACHINE_AMD64: u16 = 0x8664;
const MAGIC_PE32_PLUS: u16 = 0x20B;
const IMAGE_FILE_DLL: u16 = 0x2000;

// IMAGE_FILE_HEADER fields (offsets from the file header).
const FH_MACHINE: usize = 0;
const FH_NUMBER_OF_SECTIONS: usize = 2;
const FH_SIZE_OF_OPTIONAL_HEADER: usize = 16;
const FH_CHARACTERISTICS: usize = 18;

// IMAGE_OPTIONAL_HEADER64 fields (offsets from the optional header).
const OPT_MAGIC: usize = 0;
const OPT_IMAGE_BASE: usize = 24;
const OPT_SECTION_ALIGNMENT: usize = 32;
const OPT_FILE_ALIGNMENT: usize = 36;
const OPT_SIZE_OF_IMAGE: usize = 56;
const OPT_SIZE_OF_HEADERS: usize = 60;

// IMAGE_SECTION_HEADER: 40 bytes each.
const SECTION_HEADER_SIZE: usize = 40;
/// Maximum number of sections accepted by the Windows image loader.
const MAX_SECTIONS: usize = 96;
const SH_VIRTUAL_SIZE: usize = 8;
const SH_VIRTUAL_ADDRESS: usize = 12;
const SH_SIZE_OF_RAW_DATA: usize = 16;
const SH_POINTER_TO_RAW_DATA: usize = 20;
const SH_CHARACTERISTICS: usize = 36;

/// One section header, as written in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub raw_size: u32,
    pub raw_offset: u32,
    pub characteristics: u32,
}

impl Section {
    pub(crate) fn mapped_size(&self) -> u32 {
        match self.virtual_size {
            0 => self.raw_size,
            size => size,
        }
    }

    pub(crate) fn copied_size(&self) -> u32 {
        self.raw_size.min(self.mapped_size())
    }
}

/// The header fields the loader uses.
#[derive(Debug, Clone)]
pub struct Headers {
    pub characteristics: u16,
    pub image_base: u64,
    pub entry_rva: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub sections: Vec<Section>,
    pub(crate) section_alignment: u32,
    pub(crate) file_alignment: u32,
    pub(crate) section_table_end: usize,
    data: Vec<(u32, u32)>,
}

impl Headers {
    /// Reads the headers; fails like `loader._check_headers` on anything but an x86-64 PE32+.
    pub fn read(data: &[u8]) -> Result<Self, LoadError> {
        let file = layout::file_header_offset(data).ok_or_else(truncated)?;
        let signature = file
            .checked_sub(layout::NT_SIGNATURE_SIZE)
            .ok_or_else(truncated)?;
        if data.get(..2) != Some(b"MZ") || data.get(signature..file) != Some(PE_SIGNATURE) {
            return Err(LoadError::Malformed("missing MZ or PE signature".into()));
        }
        let opt = layout::optional_header_offset(data).ok_or_else(truncated)?;
        let machine = u16_at(data, file + FH_MACHINE)?;
        if machine != MACHINE_AMD64 || u16_at(data, opt + OPT_MAGIC)? != MAGIC_PE32_PLUS {
            return Err(LoadError::NotPe32Plus);
        }
        let section_table = opt
            .checked_add(usize::from(u16_at(
                data,
                file + FH_SIZE_OF_OPTIONAL_HEADER,
            )?))
            .ok_or_else(truncated)?;
        let count = usize::from(u16_at(data, file + FH_NUMBER_OF_SECTIONS)?);
        if count > MAX_SECTIONS {
            return Err(LoadError::SectionCount(count));
        }
        let table_end = section_table
            .checked_add(count * SECTION_HEADER_SIZE)
            .ok_or_else(truncated)?;
        let optional = data.get(opt..section_table).ok_or_else(truncated)?;
        let size_of_headers = u32_at(optional, OPT_SIZE_OF_HEADERS)?;
        let header_len = size_of_headers as usize;
        if table_end > header_len.min(data.len()) {
            return Err(LoadError::HeaderRange);
        }
        Ok(Headers {
            characteristics: u16_at(data, file + FH_CHARACTERISTICS)?,
            image_base: read_u64(optional, OPT_IMAGE_BASE).ok_or_else(truncated)?,
            entry_rva: u32_at(optional, layout::OPT_ADDRESS_OF_ENTRY_POINT)?,
            size_of_image: u32_at(optional, OPT_SIZE_OF_IMAGE)?,
            size_of_headers,
            section_alignment: u32_at(optional, OPT_SECTION_ALIGNMENT)?,
            file_alignment: u32_at(optional, OPT_FILE_ALIGNMENT)?,
            section_table_end: table_end,
            sections: (0..count)
                .map(|index| read_section(data, section_table + index * SECTION_HEADER_SIZE))
                .collect::<Result<_, _>>()?,
            data: read_directories(optional)?,
        })
    }

    pub fn is_dll(&self) -> bool {
        self.characteristics & IMAGE_FILE_DLL != 0
    }

    /// `(rva, size)` of data directory `index`, `(0, 0)` past `NumberOfRvaAndSizes`.
    pub fn directory(&self, index: usize) -> (u32, u32) {
        self.data.get(index).copied().unwrap_or((0, 0))
    }

    /// Present as `loader._directory` sees it: a non-zero RVA.
    pub fn has_directory(&self, index: usize) -> bool {
        self.directory(index).0 != 0
    }
}

fn read_directories(optional: &[u8]) -> Result<Vec<(u32, u32)>, LoadError> {
    let count = u32_at(optional, layout::OPT_NUMBER_OF_RVA_AND_SIZES)? as usize;
    optional[layout::OPT_DATA_DIRECTORY..]
        .chunks_exact(layout::DATA_DIRECTORY_SIZE)
        .take(count.min(MAX_DIRECTORIES))
        .map(|entry| Ok((u32_at(entry, 0)?, u32_at(entry, 4)?)))
        .collect()
}

/// The PE format defines 16 data directories; a larger `NumberOfRvaAndSizes` is ignored.
const MAX_DIRECTORIES: usize = 16;

fn read_section(data: &[u8], at: usize) -> Result<Section, LoadError> {
    Ok(Section {
        virtual_size: u32_at(data, at + SH_VIRTUAL_SIZE)?,
        virtual_address: u32_at(data, at + SH_VIRTUAL_ADDRESS)?,
        raw_size: u32_at(data, at + SH_SIZE_OF_RAW_DATA)?,
        raw_offset: u32_at(data, at + SH_POINTER_TO_RAW_DATA)?,
        characteristics: u32_at(data, at + SH_CHARACTERISTICS)?,
    })
}

fn truncated() -> LoadError {
    LoadError::Malformed("truncated PE headers".into())
}

fn u32_at(data: &[u8], at: usize) -> Result<u32, LoadError> {
    read_u32(data, at).ok_or_else(truncated)
}

fn u16_at(data: &[u8], at: usize) -> Result<u16, LoadError> {
    let end = at.checked_add(2).ok_or_else(truncated)?;
    let bytes = data.get(at..end).ok_or_else(truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}
