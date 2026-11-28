//! `read_pe` on images built from source (tests/common), the cases of tests/test_modules.py,
//! test_native_tls*.py and test_loader-style checks of the Python loader.

mod common;

use anyxboxx_pe::{
    DEFAULT_BASE, LoadError, PROT_EXEC, PROT_READ, PROT_WRITE, THUNKS, default_allocate,
    page_protections, read_pe,
};
use common::{DLL_BASE, EXE_BASE, Pe, Symbol, Target, rva};

const MOVED: u64 = 0x2_0000_0000;
const TLS_AT: usize = 0x100;
const CALLBACKS_AT: usize = 0x140;

fn load(file: &[u8]) -> Result<anyxboxx_pe::Image, LoadError> {
    read_pe(file, default_allocate)
}

fn forced(base: u64) -> impl FnOnce(Option<u64>, u64, bool) -> Result<u64, LoadError> {
    move |_, _, _| Ok(base)
}

#[test]
fn imports_name_every_iat_slot_by_name_or_ordinal() {
    let pe = Pe {
        imports: vec![
            ("kernel32.dll", Symbol::Name("ExitProcess")),
            ("xmem.dll", Symbol::Ordinal(12)),
            ("kernel32.dll", Symbol::Name("Sleep")),
        ],
        ..Pe::default()
    };
    let (file, iat) = pe.build();
    let image = load(&file).unwrap();
    let found: Vec<_> = image
        .imports
        .iter()
        .map(|i| (i.dll.as_str(), i.name.as_str(), i.slot))
        .collect();
    let slot =
        |dll: &str, name: &str| DLL_BASE + u64::from(iat[&(dll.to_string(), name.to_string())]);
    assert_eq!(
        found,
        [
            (
                "kernel32.dll",
                "ExitProcess",
                slot("kernel32.dll", "ExitProcess")
            ),
            ("kernel32.dll", "Sleep", slot("kernel32.dll", "Sleep")),
            ("xmem.dll", "#12", slot("xmem.dll", "#12")),
        ]
    );
    assert!(image.is_dll && !image.relocated);
    assert_eq!(image.entry, DLL_BASE + u64::from(rva(0)));
}

#[test]
fn exports_by_name_ordinal_and_forwarder() {
    let pe = Pe {
        exports: vec![
            ("Double", Target::Code(0x10)),
            ("Twice", Target::Forward("libB.Double")),
            ("Add1", Target::Code(0x20)),
        ],
        ..Pe::default()
    };
    let image = load(&pe.build().0).unwrap();
    let at = |offset| DLL_BASE + u64::from(rva(offset));
    assert_eq!(image.exports.get("Double"), Some(&at(0x10)));
    assert_eq!(image.exports.get("Add1"), Some(&at(0x20)));
    assert_eq!(image.ordinals.get(&1), Some(&at(0x10)));
    assert_eq!(image.ordinals.get(&3), Some(&at(0x20)));
    assert_eq!(
        image.forwarders.get("Twice").map(String::as_str),
        Some("libB.Double")
    );
    assert_eq!(
        image.forwarders.get("#2").map(String::as_str),
        Some("libB.Double")
    );
    assert!(!image.exports.contains_key("Twice"));
}

#[test]
fn a_forced_base_relocates_pointers_entry_and_exports() {
    let pe = Pe {
        pointers: vec![(0x80, 0x10)],
        exports: vec![("Double", Target::Code(0x10))],
        ..Pe::default()
    };
    let image = read_pe(&pe.build().0, forced(MOVED)).unwrap();
    assert!(image.relocated);
    assert_eq!(image.base, MOVED);
    assert_eq!(image.entry, MOVED + u64::from(rva(0)));
    assert_eq!(image.exports["Double"], MOVED + u64::from(rva(0x10)));
}

#[test]
fn an_unusable_preferred_base_falls_back_only_when_relocatable() {
    let relocatable = Pe {
        base: Some(THUNKS),
        pointers: vec![(0x80, 0)],
        ..Pe::default()
    };
    let image = load(&relocatable.build().0).unwrap();
    assert_eq!(image.base, DEFAULT_BASE);
    assert!(image.relocated);
    let fixed = Pe {
        base: Some(THUNKS),
        ..Pe::default()
    };
    assert!(matches!(
        load(&fixed.build().0),
        Err(LoadError::UnusableBase)
    ));
}

fn tls_image() -> Vec<u8> {
    let mut code = vec![0u8; CALLBACKS_AT + 24];
    // IMAGE_TLS_DIRECTORY64: template [0x180, 0x190), index at 0x1A0, callbacks at 0x140.
    code[TLS_AT + 32..TLS_AT + 36].copy_from_slice(&0x30u32.to_le_bytes());
    let pe = Pe {
        dll: false,
        code,
        tls: Some(TLS_AT),
        pointers: vec![
            (TLS_AT, 0x180),
            (TLS_AT + 8, 0x190),
            (TLS_AT + 16, 0x1A0),
            (TLS_AT + 24, CALLBACKS_AT),
            (CALLBACKS_AT, 0x40),
            (CALLBACKS_AT + 8, 0x60),
        ],
        ..Pe::default()
    };
    pe.build().0
}

#[test]
fn tls_directory_and_callbacks_follow_the_chosen_base() {
    let file = tls_image();
    let at = |base: u64, offset| base + u64::from(rva(offset));
    for base in [EXE_BASE, MOVED] {
        let tls = read_pe(&file, forced(base)).unwrap().tls.unwrap();
        assert_eq!(
            (tls.template_start, tls.template_end),
            (at(base, 0x180), at(base, 0x190))
        );
        assert_eq!(tls.zero_fill, 0x30);
        assert_eq!(tls.index_address, at(base, 0x1A0));
        assert_eq!(tls.callbacks, [at(base, 0x40), at(base, 0x60)]);
    }
}

#[test]
fn foreign_or_unsupported_images_are_refused() {
    let (file, _) = Pe {
        machine: 0x14C,
        ..Pe::default()
    }
    .build();
    assert!(matches!(load(&file), Err(LoadError::NotPe32Plus)));
    let (file, _) = Pe {
        dll: false,
        entry: None,
        ..Pe::default()
    }
    .build();
    assert!(matches!(load(&file), Err(LoadError::BadHeaders)));
    let (mut file, _) = Pe::default().build();
    Pe::set_directory(&mut file, 13, 0x1000, 0x40);
    assert!(matches!(load(&file), Err(LoadError::Unsupported(_))));
    let (mut file, _) = Pe::default().build();
    Pe::set_directory(&mut file, 14, 0x1000, 0x48);
    assert!(matches!(load(&file), Err(LoadError::Unsupported(_))));
    assert!(load(b"MZ not a PE").is_err());
    assert!(load(&[]).is_err());
}

#[test]
fn a_dll_without_entry_point_is_accepted() {
    let image = load(
        &Pe {
            entry: None,
            ..Pe::default()
        }
        .build()
        .0,
    )
    .unwrap();
    assert_eq!(image.entry, 0);
}

#[test]
fn pages_take_the_section_rights() {
    let (file, _) = Pe::default().build();
    let pages = page_protections(&file).unwrap();
    assert_eq!(pages[0], PROT_READ);
    assert!(
        pages[1..]
            .iter()
            .all(|&p| p == PROT_READ | PROT_WRITE | PROT_EXEC)
    );
}

#[test]
fn a_malformed_relocation_block_is_refused_not_skipped() {
    const DIRECTORIES: usize = 0x80 + 4 + 20 + 112;
    const DIR_RELOC: usize = 5;
    const HEADERS_SIZE: usize = 0x400;
    let pe = Pe {
        pointers: vec![(0x80, 0x10)],
        ..Pe::default()
    };
    let (mut file, _) = pe.build();
    let at = DIRECTORIES + 8 * DIR_RELOC;
    let table_rva = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
    let block = table_rva - common::SECTION_RVA as usize + HEADERS_SIZE;
    // SizeOfBlock smaller than the block header: pefile stopped there and left the pointer as is.
    file[block + 4..block + 8].copy_from_slice(&6u32.to_le_bytes());
    assert!(matches!(
        read_pe(&file, forced(MOVED)),
        Err(LoadError::Relocation(_))
    ));
}

const COFF: usize = 0x80 + 4;
const OPTIONAL: usize = COFF + 20;
const SECTION: usize = OPTIONAL + 240;
const IMAGE_SIZE: usize = OPTIONAL + 56;
const HEADER_SIZE: usize = OPTIONAL + 60;

#[path = "read_pe/layout_compat.rs"]
mod layout_compat;

fn put_u32(file: &mut [u8], offset: usize, value: u32) {
    file[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn refuse_before_base_allocation(file: &[u8]) -> LoadError {
    read_pe(file, |_, _, _| {
        panic!("malformed layout reached the base allocator")
    })
    .unwrap_err()
}

#[test]
fn image_dimensions_are_bounded_on_both_entry_points() {
    let (mut file, _) = Pe::default().build();
    put_u32(&mut file, IMAGE_SIZE, 0);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::BadHeaders
    ));

    let maximum = anyxboxx_pe::MAX_IMAGE;
    put_u32(&mut file, IMAGE_SIZE, maximum as u32);
    let pages = page_protections(&file).unwrap();
    assert_eq!(pages.len() as u64, maximum / anyxboxx_pe::PAGE);
    assert_eq!(pages[0], PROT_READ);
    assert_eq!(pages[1], PROT_READ | PROT_WRITE | PROT_EXEC);
    assert_eq!(pages[2], PROT_READ);

    put_u32(&mut file, IMAGE_SIZE, (maximum + anyxboxx_pe::PAGE) as u32);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::TooLarge
    ));
    assert!(matches!(page_protections(&file), Err(LoadError::TooLarge)));
}

#[test]
fn header_size_and_section_count_are_checked() {
    let (mut file, _) = Pe::default().build();
    put_u32(&mut file, HEADER_SIZE, SECTION as u32);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::HeaderRange
    ));

    let (mut file, _) = Pe::default().build();
    file[COFF + 2..COFF + 4].copy_from_slice(&97u16.to_le_bytes());
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::SectionCount(97)
    ));
}

#[test]
fn raw_truncation_and_ranges_that_wrap_u32_are_rejected() {
    let (mut file, _) = Pe::default().build();
    file.pop();
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::RawSection(0x1000)
    ));
    assert!(matches!(
        page_protections(&file),
        Err(LoadError::RawSection(0x1000))
    ));

    let (mut file, _) = Pe::default().build();
    put_u32(&mut file, SECTION + 20, 0xFFFF_FE00);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::RawSection(0x1000)
    ));
    put_u32(&mut file, SECTION + 12, 0xFFFF_F000);
    put_u32(&mut file, SECTION + 8, 0x2000);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::Section(0xFFFF_F000)
    ));
}

#[test]
fn virtual_overlap_and_malformed_alignment_are_rejected() {
    let (mut file, _) = Pe::default().build();
    file[COFF + 2..COFF + 4].copy_from_slice(&2u16.to_le_bytes());
    put_u32(&mut file, SECTION + 40 + 8, 0x100);
    put_u32(&mut file, SECTION + 40 + 12, 0x1000);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::Section(0x1000)
    ));

    let (mut file, _) = Pe::default().build();
    put_u32(&mut file, OPTIONAL + 36, 3);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::Alignment
    ));
}

#[test]
fn data_directories_are_bounded_in_their_address_space() {
    let (mut file, _) = Pe::default().build();
    file.resize(0x2110, 0);
    Pe::set_directory(&mut file, 4, 0x2100, 16);
    load(&file).expect("a certificate table uses file offsets outside the mapped image");
    Pe::set_directory(&mut file, 4, 0x2100, 17);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::DirectoryRange(4)
    ));
    Pe::set_directory(&mut file, 4, 0, 0);
    Pe::set_directory(&mut file, 1, u32::MAX - 1, 8);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::DirectoryRange(1)
    ));
}

#[test]
fn relocation_target_range_is_checked_before_base_allocation() {
    let (mut file, _) = Pe {
        pointers: vec![(0x80, 0x10)],
        ..Pe::default()
    }
    .build();
    let header = anyxboxx_pe::Headers::read(&file).unwrap();
    let (rva, _) = header.directory(5);
    let section = header.sections[0];
    let reloc = (section.raw_offset + rva - section.virtual_address) as usize;
    put_u32(&mut file, reloc, 0xFFFF_F000);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::Relocation(_)
    ));
}

#[test]
fn a_relocation_table_across_two_sections_can_be_mapped() {
    let (mut file, _) = Pe::default().build();
    file.resize(0x1600, 0);
    file[COFF + 2..COFF + 4].copy_from_slice(&2u16.to_le_bytes());
    put_u32(&mut file, IMAGE_SIZE, 0x3000);
    put_u32(&mut file, SECTION + 8, 0x1000);
    put_u32(&mut file, SECTION + 16, 0x1000);
    put_u32(&mut file, SECTION + 40 + 8, 0x200);
    put_u32(&mut file, SECTION + 40 + 12, 0x2000);
    put_u32(&mut file, SECTION + 40 + 16, 0x200);
    put_u32(&mut file, SECTION + 40 + 20, 0x1400);
    Pe::set_directory(&mut file, 5, 0x1FFC, 12);
    put_u32(&mut file, 0x13FC, 0x1000);
    put_u32(&mut file, 0x1400, 12);
    file[0x1404..0x1406].copy_from_slice(&0xA008u16.to_le_bytes());
    assert!(read_pe(&file, forced(MOVED)).unwrap().relocated);
}
