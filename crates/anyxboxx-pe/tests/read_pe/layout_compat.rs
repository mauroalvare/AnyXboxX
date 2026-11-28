use super::*;

#[test]
fn raw_padding_does_not_extend_the_mapped_section() {
    let (mut file, _) = Pe::default().build();
    file.resize(0x1800, 0);
    file[COFF + 2..COFF + 4].copy_from_slice(&2u16.to_le_bytes());
    put_u32(&mut file, IMAGE_SIZE, 0x3000);
    put_u32(&mut file, SECTION + 8, 0x1000);
    put_u32(&mut file, SECTION + 16, 0x1200);
    let next = SECTION + 40;
    put_u32(&mut file, next + 8, 0);
    put_u32(&mut file, next + 12, 0x2000);
    put_u32(&mut file, next + 16, 0x200);
    put_u32(&mut file, next + 20, 0x1600);
    put_u32(&mut file, next + 36, 0x4000_0000);

    let image = load(&file).unwrap();
    assert_eq!(image.size, 0x3000);
    assert_eq!(
        page_protections(&file).unwrap(),
        [PROT_READ, PROT_READ | PROT_WRITE | PROT_EXEC, PROT_READ]
    );
}

#[test]
fn unaligned_raw_data_and_oversized_header_hints_keep_the_same_imports() {
    let (mut file, _) = Pe {
        imports: vec![("fixture.dll", Symbol::Name("entry"))],
        ..Pe::default()
    }
    .build();
    let expected = load(&file).unwrap();
    let raw = anyxboxx_pe::Headers::read(&file).unwrap().sections[0].raw_offset;
    file.insert(raw as usize, 0);
    put_u32(&mut file, SECTION + 20, raw + 1);
    put_u32(&mut file, HEADER_SIZE, raw + 1);
    assert_eq!(load(&file).unwrap(), expected);

    file.resize(0x1400, 0);
    let overstated_header = file.len() as u32 + 64;
    put_u32(&mut file, HEADER_SIZE, overstated_header);
    assert_eq!(load(&file).unwrap(), expected);
}

#[test]
fn unused_and_absent_directories_do_not_gate_loading() {
    let (mut file, _) = Pe::default().build();
    let expected = load(&file).unwrap();
    Pe::set_directory(&mut file, 5, 0, 0x100);
    Pe::set_directory(&mut file, 10, u32::MAX, 0x100);
    assert_eq!(load(&file).unwrap(), expected);
}

#[test]
fn bound_imports_are_checked_against_file_bytes() {
    let (mut file, _) = Pe::default().build();
    file.resize(0x2110, 0);
    Pe::set_directory(&mut file, 11, 0x2100, 16);
    load(&file).unwrap();
    Pe::set_directory(&mut file, 11, 0x2100, 17);
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::DirectoryRange(11)
    ));
}

#[test]
fn header_inspection_reads_only_complete_directory_entries() {
    let (mut file, _) = Pe::default().build();
    put_u32(&mut file, OPTIONAL + 108, 17);
    assert!(page_protections(&file).is_ok());
    // The shared PE parser still requires the declared directory array to fit
    // inside the optional header before it reads imports and exports.
    assert!(matches!(
        refuse_before_base_allocation(&file),
        LoadError::Pe(_)
    ));
}
