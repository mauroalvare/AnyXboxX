"""Inventory static call sites and embedded shader-container metadata in the three graphics DLLs."""

import argparse
from collections import defaultdict
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import pefile

from pe_inventory import EXPERIMENTS
from resource_inventory import dxil_program, parts as container_parts

NAMES = ("Bink2Scarlett.dll", "XboxBvhBuilder.dll", "amd_fidelityfx_upscaler_xbox.dll")
INSTRUCTION = re.compile(r"^\s*([\dA-F]{16}):\s+(\w+)\s+(.*)$", re.I)
ADDRESS = re.compile(r"\[([0-9a-f]{16})h\]", re.I)


def instruction_refs(path, base):
    output = subprocess.check_output(["dumpbin", "/nologo", "/disasm:nobytes", str(path)])
    memory, branches = defaultdict(list), defaultdict(list)
    for line in output.decode("ascii", "replace").splitlines():
        match = INSTRUCTION.match(line)
        if not match:
            continue
        va, op, operands = match.groups()
        rva = int(va, 16) - base
        target = ADDRESS.search(operands)
        if target:
            memory[int(target[1], 16) - base].append((rva, op))
        if op in ("call", "jmp") and re.fullmatch("[0-9a-fA-F]{16}", operands):
            branches[int(operands, 16) - base].append((rva, op))
    return memory, branches


def _mapped_rva(pe, offset, length):
    section = pe.get_section_by_offset(offset)
    if section is None or not section.PointerToRawData:
        return None
    rel = offset - section.PointerToRawData
    limit = min(section.SizeOfRawData, section.Misc_VirtualSize) if section.Misc_VirtualSize else section.SizeOfRawData
    if rel < 0 or rel + length > limit:
        return None
    return section.VirtualAddress + rel


def shader_containers(data, pe, memory):
    base = pe.OPTIONAL_HEADER.ImageBase
    rows = []
    for marker in re.finditer(b"DXBC", data):
        off = marker.start()
        if off + 32 > len(data):
            continue
        version, size, count = struct.unpack_from("<3I", data, off + 20)
        if version != 1 or not 1 <= count <= 128 or not 32 + count * 4 <= size <= len(data) - off:
            continue
        try:
            sections = container_parts(data[off:off + size])
        except ValueError:
            continue
        rva = _mapped_rva(pe, off, size)
        if rva is None:
            continue
        root = None
        rts0 = sections.get(b"RTS0")
        if rts0 is not None and len(rts0) >= 24:
            ver, nparam, _, nsampler, _, flags = struct.unpack_from("<6I", rts0)
            root = dict(version=ver, parameter_count=nparam, static_sampler_count=nsampler, flags=hex(flags))
        refs = []
        for ref in re.finditer(re.escape(struct.pack("<Q", base + rva)), data):
            pos = ref.start()
            if pos >= 8 and struct.unpack_from("<I", data, pos - 8)[0] == size:
                size_slot = _mapped_rva(pe, pos - 8, 16)
                if size_slot is None:
                    continue
                slot = size_slot + 8
                refs.append(dict(pointer_slot=hex(slot), size_slot=hex(size_slot),
                                 pointer_code_refs=[hex(a) for a, _ in memory[slot]],
                                 size_code_refs=[hex(a) for a, _ in memory[size_slot]]))
        rows.append(dict(rva=hex(rva), bytes=size,
                         parts=[tag.decode("ascii", "replace") for tag in sections],
                         root_header=root, framed_dxil=dxil_program(data, off),
                         size_pointer_pairs=refs,
                         direct_code_refs=[hex(a) for a, _ in memory[rva]]))
    return rows


def inspect(path):
    data = path.read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        pe.parse_data_directories(directories=[1])
        base = pe.OPTIONAL_HEADER.ImageBase
        memory, branches = instruction_refs(path, base)
        calls = {}
        for desc in getattr(pe, "DIRECTORY_ENTRY_IMPORT", None) or ():
            dll = desc.dll.decode("ascii", "replace").lower() if desc.dll else ""
            for imp in getattr(desc, "imports", None) or ():
                name = imp.name.decode("ascii", "replace") if imp.name else ""
                if (dll, name) not in EXPERIMENTS:
                    continue
                slot = imp.address - base
                sites = []
                for site, op in memory[slot]:
                    if op in ("call", "jmp"):
                        sites.append(dict(rva=hex(site), operation=op,
                                          upstream=[hex(a) for a, _ in branches[site]] if op == "jmp" else []))
                calls[name] = dict(iat_rva=hex(slot), sites=sites)
        return dict(sha256=hashlib.sha256(data).hexdigest(), calls=calls,
                    containers=shader_containers(data, pe, memory))


def _validate_inputs(root: Path, parser: argparse.ArgumentParser) -> None:
    missing = []
    for name in NAMES:
        image = root / name
        if image.is_symlink():
            raise ValueError(f"Expected image must not be a symlink: {image}")
        if not image.exists():
            missing.append(name)
            continue
        if image.is_dir():
            raise IsADirectoryError(f"Expected image path is a directory: {image}")
        if not image.is_file():
            raise ValueError(f"Expected image must be a regular file: {image}")
        if not os.access(image, os.R_OK):
            raise PermissionError(f"Expected image is not readable: {image}")
        with image.open("rb"):
            pass
    if missing:
        parser.error(f"Missing expected image(s) in {root}: {', '.join(missing)}")


def _validate_output(output: Path) -> None:
    if output.is_symlink():
        raise ValueError(f"Output path must not be a symlink: {output}")
    if output.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output}")
    if output.exists() and not output.is_file():
        raise ValueError(f"Output path is not a regular file: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"Output parent is not a directory: {parent}")


def _write_atomic(path: Path, content: str) -> None:
    """Write UTF-8 text to a sibling temp file and atomically replace path, leaving 0 temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.directory.resolve(strict=True)
    except OSError as exc:
        parser.error(str(exc))
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    output = Path(__file__).resolve().parents[1] / "out/calls/six-apis.json"
    try:
        if output.is_relative_to(root) or output.resolve().is_relative_to(root):
            parser.error("Choose a build directory separate from the report directory")
        _validate_output(output)
        _validate_inputs(root, parser)
        rows = {name: inspect(root / name) for name in NAMES}
        _write_atomic(output, json.dumps(rows, indent=2) + "\n")
    except (OSError, ValueError, struct.error, pefile.PEFormatError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"call_evidence: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
