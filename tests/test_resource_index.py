"""Synthetic inputs for the versioned range sidecar and its direct-read fallback."""

import hashlib
import json
import os
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

from tools.resource_index import (
    Codec, Confidence, IndexFormatError, PublishError, RangeError, ResourceRange,
    SourceChanged, build_index, open_resource,
)
from tools.resource_index import index as implementation
from tools.resource_index.__main__ import main
from tools.resource_index.format import (
    HEADER, MAX_EXPANSION, RECORD, U64_MAX, decode_range, encode_range, validate_range,
)

SOURCE = Path("source.bin")
DATA = bytes(range(32))
FIRST = ResourceRange(2, 8, 0, 8, 8, Codec.RAW, Confidence.CONFIRMED)
SECOND = ResourceRange(10, 4, 12, 7, 16, Codec.ZLIB, Confidence.DECLARED)


@pytest.fixture
def files(tmp_path):
    (tmp_path / SOURCE).write_bytes(DATA)
    return tmp_path, tmp_path / "source.axxi"


def test_deterministic_indexes_and_exact_reads(files):
    root, index_path = files
    build_index(root, SOURCE, iter([FIRST, SECOND]), index_path)
    first = index_path.read_bytes()
    build_index(root, SOURCE, iter([FIRST, SECOND]), index_path)
    assert index_path.read_bytes() == first
    assert HEADER.unpack_from(first)[-1] == hashlib.sha256(DATA).digest()

    with open_resource(root, SOURCE, index_path, strict=True) as (source, index):
        assert index.lookup(0, 8) == FIRST
        assert index.lookup(13, 5) == SECOND
        assert index.lookup(7, 2) is None
        assert index.lookup(8, 1) is None
        assert index.lookup(19, 1) is None
        hint = index.lookup(3, 4)
        source.seek(hint.physical_offset + (3 - hint.logical_offset))
        assert source.read(4) == DATA[5:9]
    assert (root / SOURCE).read_bytes() == DATA


@pytest.mark.parametrize("item", [
    replace(FIRST, physical_offset=U64_MAX, physical_length=2),
    replace(FIRST, logical_offset=U64_MAX - FIRST.logical_length + 1),
    replace(FIRST, physical_offset=26),
    replace(FIRST, expansion_limit=MAX_EXPANSION + 1),
    replace(FIRST, confidence=0),
    replace(FIRST, logical_length=0),
    replace(FIRST, physical_offset=True),
])
def test_rejects_invalid_intervals_before_publishing(files, item):
    root, index_path = files
    with pytest.raises(RangeError):
        build_index(root, SOURCE, [item], index_path)
    assert not index_path.exists()
    assert not list(root.glob(".resource-index-*"))


def test_offsets_are_wide_in_both_address_spaces():
    item = replace(FIRST, physical_offset=1 << 40, logical_offset=(1 << 50) + 3)
    validate_range(item, (1 << 40) + item.physical_length, None)
    assert decode_range(encode_range(item)) == item


@pytest.mark.parametrize("second", [
    replace(SECOND, physical_offset=9),
    replace(SECOND, logical_offset=7),
    replace(SECOND, physical_offset=0, logical_offset=0),
])
def test_ambiguous_bounds_leave_the_previous_index(files, second):
    root, index_path = files
    build_index(root, SOURCE, [FIRST], index_path)
    first = index_path.read_bytes()
    with pytest.raises(RangeError, match="non-overlapping"):
        build_index(root, SOURCE, [FIRST, second], index_path)
    assert first == index_path.read_bytes()
    assert not list(root.glob(".resource-index-*"))


@pytest.mark.parametrize("mutation", ["schema", "parser", "truncated", "extra", "checksum", "count", "ambiguous"])
def test_consumer_rejects_corrupt_or_ambiguous_metadata(files, mutation):
    root, index_path = files
    build_index(root, SOURCE, [FIRST, SECOND], index_path)
    data = bytearray(index_path.read_bytes())
    if mutation == "schema":
        data[8] = 2
    elif mutation == "parser":
        data[10] = 2
    elif mutation == "truncated":
        del data[HEADER.size + RECORD.size + 1:]
    elif mutation == "extra":
        data.append(0)
    elif mutation == "checksum":
        data[-1] ^= 1
    elif mutation == "count":
        data[20:28] = U64_MAX.to_bytes(8, "little")
    else:
        data[HEADER.size + RECORD.size:-32] = encode_range(replace(SECOND, physical_offset=9))
        data[-32:] = hashlib.sha256(data[HEADER.size:-32]).digest()
    index_path.write_bytes(data)
    with pytest.raises(IndexFormatError), open_resource(root, SOURCE, index_path, strict=True):
        pytest.fail("the corrupt sidecar was accepted")
    with open_resource(root, SOURCE, index_path) as (source, index):
        assert index is None
        assert source.read() == DATA


def test_same_name_and_length_with_new_hash_invalidates_index(files):
    root, index_path = files
    build_index(root, SOURCE, [FIRST], index_path)
    new_data = DATA[::-1]
    (root / SOURCE).write_bytes(new_data)
    with pytest.raises(SourceChanged, match="SHA-256"), open_resource(root, SOURCE, index_path, strict=True):
        pytest.fail("the stale sidecar was accepted")
    for sidecar in (index_path, None, root / "missing.axxi"):
        with open_resource(root, SOURCE, sidecar) as (source, index):
            assert index is None
            assert source.read() == new_data


def test_failed_source_hash_preserves_direct_read_cursor(files, monkeypatch):
    root, index_path = files
    build_index(root, SOURCE, [FIRST], index_path)

    def interrupted_hash(source, _algorithm):
        source.read(4)
        raise OSError("synthetic read interruption")

    monkeypatch.setattr(implementation.hashlib, "file_digest", interrupted_hash)
    with open_resource(root, SOURCE, index_path) as (source, index):
        assert index is None
        assert source.read() == DATA


def test_change_during_build_and_failed_replace_keep_the_published_file(files, monkeypatch):
    root, index_path = files
    build_index(root, SOURCE, [FIRST], index_path)
    old = index_path.read_bytes()

    def ranges():
        yield FIRST
        source = root / SOURCE
        info = source.stat()
        source.write_bytes(DATA[::-1])
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))

    with pytest.raises(SourceChanged):
        build_index(root, SOURCE, ranges(), index_path)
    assert index_path.read_bytes() == old

    def fail_replace(_temporary, _index):
        raise PermissionError("synthetic publication failure")

    monkeypatch.setattr(implementation.os, "replace", fail_replace)
    with pytest.raises(PublishError, match="synthetic publication failure"):
        build_index(root, SOURCE, [SECOND], index_path)
    assert index_path.read_bytes() == old
    assert not list(root.glob(".resource-index-*"))


@pytest.mark.parametrize("changed", ["index", "source"])
def test_changes_on_open_handles_invalidate_hints(files, changed):
    root, index_path = files
    build_index(root, SOURCE, [FIRST], index_path)
    with open_resource(root, SOURCE, index_path, strict=True) as (_, index):
        assert index.lookup(0, 1) == FIRST
        path = index_path if changed == "index" else root / SOURCE
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        assert index.lookup(0, 1) is None


def test_source_root_and_destination_are_checked(files):
    root, index_path = files
    child = root / "authorized"
    child.mkdir()
    with pytest.raises(IndexFormatError, match="escapes"):
        build_index(child, Path("..") / SOURCE, [FIRST], index_path)
    with pytest.raises(PublishError, match="replace the source"):
        build_index(root, SOURCE, [FIRST], root / SOURCE)
    assert (root / SOURCE).read_bytes() == DATA


def test_build_and_validation_use_fixed_memory(files):
    root, index_path = files
    count = 100_000
    (root / SOURCE).write_bytes(b"f" * count)
    tracemalloc.start()
    try:
        build_index(
            root, SOURCE,
            (ResourceRange(n, 1, n + (1 << 32), 1, 1, Codec.RAW, Confidence.CONFIRMED) for n in range(count)),
            index_path,
        )
        with open_resource(root, SOURCE, index_path, strict=True) as (_, index):
            assert index.lookup((1 << 32) + count - 1, 1).physical_offset == count - 1
        assert tracemalloc.get_traced_memory()[1] < 2 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_cli_round_trip_and_bounds_on_json_lines(files, capsys):
    root, index_path = files
    metadata = root / "ranges.jsonl"
    metadata.write_text(json.dumps({
        "physical_offset": 2, "physical_length": 8, "logical_offset": 0,
        "logical_length": 8, "expansion_limit": 8, "codec": "raw", "confidence": "confirmed",
    }) + "\n", encoding="utf-8")
    base = ["--root", str(root), "--source", str(SOURCE), "--index", str(index_path)]
    main([*base, "build", str(metadata)])
    main([*base, "query", "0", "8"])
    assert json.loads(capsys.readouterr().out)["physical_offset"] == 2
    old = index_path.read_bytes()
    metadata.write_bytes(b" " * 4097)
    with pytest.raises(SystemExit) as failure:
        main([*base, "build", str(metadata)])
    assert failure.value.code == 1
    assert index_path.read_bytes() == old
