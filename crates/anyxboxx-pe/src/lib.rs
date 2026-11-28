//! AnyXboxX's view of a PE32+ x86-64 image: where it is placed, what it imports and exports, its
//! TLS directory and the protection of its pages.
//!
//! Rust port of `PrototypeScarlett/emu/loader.py` (`read_pe`), first component of the move to
//! Rust (docs/architecture-rust.md). The tables are parsed by `victory_pe` (reused, not
//! re-implemented); this crate adds AnyXboxX's rules on top: the checks, the choice of base,
//! relocation and the TLS template. `tests/test_rust_pe_parity.py` compares its output with
//! the Python loader on images built from source.

use std::collections::BTreeMap;

use victory_pe::ImageInfo;

mod headers;
pub mod json;
mod mapping;
mod tls;

pub use headers::{Headers, Section};
pub use mapping::{PROT_EXEC, PROT_READ, PROT_WRITE};
pub use tls::{MAX_TLS_CALLBACKS, MAX_TLS_SIZE};

pub const PAGE: u64 = 0x1000;
pub const MAX_IMAGE: u64 = 256 * 1024 * 1024;
/// Base used when the preferred one is unusable and the image can be relocated.
pub const DEFAULT_BASE: u64 = 0x1_4000_0000;
/// Areas the emulator reserves (`emu/memory.py`): an image must end before `THUNKS`.
pub const THUNKS: u64 = 0x7000_0000_0000;
pub const HEAP: u64 = 0x7000_1000_0000;

const DIR_IMPORT_DELAY: usize = 13;
const DIR_CLR: usize = 14;

/// Errors of [`read_pe`]: every refusal is explicit, nothing is silently skipped.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("image larger than 256 MiB")]
    TooLarge,
    #[error("the program must be an x86-64 PE32+ executable")]
    NotPe32Plus,
    #[error("incompatible PE image or missing entry point")]
    BadHeaders,
    #[error("PE section table or declared headers exceed the file header range")]
    HeaderRange,
    #[error("unsupported PE section count {0}")]
    SectionCount(usize),
    #[error("incompatible PE file or section alignment")]
    Alignment,
    #[error("PE data directory {0} is outside its valid range")]
    DirectoryRange(usize),
    #[error("PE section at RVA {0:#x} has an invalid raw range")]
    RawSection(u64),
    #[error("could not reserve {0} bytes for PE mapping data")]
    Allocation(usize),
    #[error("feature not emulated: {0}")]
    Unsupported(&'static str),
    #[error("malformed image: {0}")]
    Malformed(String),
    #[error("unusable image base and no readable relocation table")]
    UnusableBase,
    #[error("import {0} outside the image")]
    ImportOutside(String),
    #[error("{0}")]
    Tls(String),
    #[error("incompatible PE section at RVA {0:#x}")]
    Section(u64),
    #[error("relocation: {0}")]
    Relocation(String),
    #[error(transparent)]
    Pe(#[from] victory_pe::PeError),
}

/// One IAT slot to fill: `name` is `#n` for an import by ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub dll: String,
    pub name: String,
    pub slot: u64,
}

/// TLS directory, with addresses at the chosen base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tls {
    pub template_start: u64,
    pub template_end: u64,
    pub zero_fill: u32,
    pub index_address: u64,
    pub callbacks: Vec<u64>,
}

/// The loaded image, as `loader.Image`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub base: u64,
    pub size: u64,
    pub entry: u64,
    pub imports: Vec<Import>,
    pub exports: BTreeMap<String, u64>,
    pub ordinals: BTreeMap<u16, u64>,
    /// Name or `#n` → `DLL.Function`.
    pub forwarders: BTreeMap<String, String>,
    pub tls: Option<Tls>,
    pub relocated: bool,
    pub is_dll: bool,
}

pub fn align(size: u64) -> u64 {
    size.div_ceil(PAGE) * PAGE
}

/// Placement of a lone image: the preferred base, else `DEFAULT_BASE` if relocatable.
pub fn default_allocate(
    preferred: Option<u64>,
    _size: u64,
    relocatable: bool,
) -> Result<u64, LoadError> {
    match preferred {
        Some(base) => Ok(base),
        None if relocatable => Ok(DEFAULT_BASE),
        None => Err(LoadError::UnusableBase),
    }
}

/// Analyses an image. `allocate(preferred base or None, aligned size, relocatable)` picks the
/// final base; a preferred base overlapping the emulator's areas is passed as `None`.
pub fn read_pe<F>(data: &[u8], allocate: F) -> Result<Image, LoadError>
where
    F: FnOnce(Option<u64>, u64, bool) -> Result<u64, LoadError>,
{
    if data.len() as u64 > MAX_IMAGE {
        return Err(LoadError::TooLarge);
    }
    let headers = Headers::read(data)?;
    check(&headers)?;
    let layout = mapping::ImageLayout::check(data, &headers)?;
    let relocations = layout.relocations()?;
    let info = ImageInfo::parse(data)?;
    let mut mapped = layout.map()?;
    let size = layout.size() as u64;
    let base = allocate(
        usable(headers.image_base, size),
        size,
        relocations.has_entries(),
    )?;
    let delta = base.wrapping_sub(headers.image_base);
    let imports = imports(&info, base, size)?;
    let tls_fields = tls::fields(&mapped, &headers, delta)?;
    relocations.apply(&mut mapped, delta)?;
    let (exports, ordinals, forwarders) = exports(&info, base);
    Ok(Image {
        base,
        size,
        entry: if headers.entry_rva == 0 {
            0
        } else {
            base + u64::from(headers.entry_rva)
        },
        imports,
        exports,
        ordinals,
        forwarders,
        tls: tls_fields
            .map(|fields| tls::resolve(fields, &mapped, base))
            .transpose()?,
        relocated: delta != 0,
        is_dll: headers.is_dll(),
    })
}

/// Protection of each page of the image (`PROT_*` bits), as `loader._map` sets them.
pub fn page_protections(data: &[u8]) -> Result<Vec<u8>, LoadError> {
    mapping::ImageLayout::check(data, &Headers::read(data)?)?.page_protections()
}

fn check(headers: &Headers) -> Result<(), LoadError> {
    let size = headers.size_of_image;
    let entry = headers.entry_rva;
    let entry_ok = entry < size && (headers.is_dll() || entry > 0);
    if !entry_ok {
        return Err(LoadError::BadHeaders);
    }
    if headers.has_directory(DIR_IMPORT_DELAY) {
        return Err(LoadError::Unsupported("delay-load imports"));
    }
    if headers.has_directory(DIR_CLR) {
        return Err(LoadError::Unsupported(".NET"));
    }
    Ok(())
}

fn usable(base: u64, size: u64) -> Option<u64> {
    let bad = !base.is_multiple_of(PAGE)
        || base == 0
        || base.saturating_add(size) > THUNKS
        || base >= HEAP;
    (!bad).then_some(base)
}

fn imports(info: &ImageInfo, base: u64, size: u64) -> Result<Vec<Import>, LoadError> {
    let mut found = Vec::new();
    for module in &info.imports {
        for entry in &module.entries {
            let slot = base + u64::from(entry.iat_slot_rva);
            let name = entry.symbol.to_string();
            if u64::from(entry.iat_slot_rva) + 8 > size {
                return Err(LoadError::ImportOutside(format!("{}!{name}", module.dll)));
            }
            found.push(Import {
                dll: module.dll.clone(),
                name,
                slot,
            });
        }
    }
    Ok(found)
}

type Exports = (
    BTreeMap<String, u64>,
    BTreeMap<u16, u64>,
    BTreeMap<String, String>,
);

fn exports(info: &ImageInfo, base: u64) -> Exports {
    let (mut names, mut ordinals, mut forwarders) = Exports::default();
    for export in &info.exports {
        if let Some(target) = &export.forward {
            forwarders.insert(format!("#{}", export.ordinal), target.clone());
            if let Some(name) = &export.name {
                forwarders.insert(name.clone(), target.clone());
            }
        } else if let Some(rva) = export.rva.filter(|&rva| rva != 0) {
            let address = base + u64::from(rva);
            ordinals.insert(export.ordinal, address);
            if let Some(name) = &export.name {
                names.insert(name.clone(), address);
            }
        }
    }
    (names, ordinals, forwarders)
}
