"""Publish checked ranges and look them up against the same open source handle."""

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .format import (
    DIGEST_BYTES, HEADER, MAGIC, MAX_RECORDS, PARSER_VERSION, RECORD, SCHEMA, U64_MAX,
    IndexFormatError, ResourceRange, checked_end, decode_range, encode_range, validate_range,
)


class SourceChanged(IndexFormatError):
    """The source changed identity or no longer matches the sidecar."""


class PublishError(OSError):
    """The destination could not be published atomically."""


@dataclass(frozen=True, slots=True)
class _Stamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> "_Stamp":
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    stream: BinaryIO
    stamp: _Stamp

    def check(self) -> None:
        try:
            named = self.path.stat()
        except FileNotFoundError as error:
            raise SourceChanged("source path disappeared") from error
        # Windows can expose different ctime values for path and handle metadata.
        same_file = (named.st_dev, named.st_ino) == (self.stamp.device, self.stamp.inode)
        if not same_file or _Stamp.of(os.fstat(self.stream.fileno())) != self.stamp:
            raise SourceChanged("source changed while the index was in use")

    def identity(self) -> bytes:
        self.check()
        self.stream.seek(0)
        try:
            digest = hashlib.file_digest(self.stream, "sha256").digest()
        finally:
            self.stream.seek(0)
        self.check()
        return digest


def _open_regular(path: Path) -> BinaryIO:
    if not stat.S_ISREG(path.stat().st_mode):
        raise IndexFormatError("expected a regular file")
    stream = path.open("rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise IndexFormatError("expected a regular file")
    return stream


@contextmanager
def _open_source(root: Path, relative: Path) -> Iterator[_Source]:
    root = root.resolve(strict=True)
    if relative.anchor:
        raise IndexFormatError("source path must be relative to the authorized root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise IndexFormatError("source path escapes the authorized root")
    with _open_regular(path) as stream:
        if path.resolve(strict=True) != path:
            raise SourceChanged("resolved source path changed during open")
        stamp = _Stamp.of(os.fstat(stream.fileno()))
        if not 0 <= stamp.size <= U64_MAX:
            raise IndexFormatError("source length exceeds unsigned 64-bit bounds")
        source = _Source(path, stream, stamp)
        source.check()
        yield source


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise IndexFormatError("truncated index")
    return data


def _header(file: BinaryIO, source_size: int) -> tuple[int, bytes]:
    file.seek(0)
    magic, schema, parser, size, count, digest = HEADER.unpack(_read_exact(file, HEADER.size))
    if (magic, schema, parser) != (MAGIC, SCHEMA, PARSER_VERSION):
        raise IndexFormatError("unsupported index schema or parser version")
    if count > MAX_RECORDS:
        raise IndexFormatError("record count exceeds the index limit")
    if os.fstat(file.fileno()).st_size != HEADER.size + count * RECORD.size + DIGEST_BYTES:
        raise IndexFormatError("index length does not match the record count")
    if size != source_size:
        raise SourceChanged("source length does not match the index")
    return count, digest


def _validate_records(file: BinaryIO, count: int, source_size: int) -> None:
    records_digest = hashlib.sha256()
    previous = None
    for _ in range(count):
        stored = _read_exact(file, RECORD.size)
        item = decode_range(stored)
        validate_range(item, source_size, previous)
        records_digest.update(stored)
        previous = item
    if _read_exact(file, DIGEST_BYTES) != records_digest.digest():
        raise IndexFormatError("record checksum mismatch")


def build_index(root: Path, source_path: Path, ranges: Iterable[ResourceRange], destination: Path) -> None:
    """Build an index for a source below root and publish the complete file in one replace."""
    with _open_source(root, source_path) as source:
        if destination.resolve() == source.path or (
            destination.exists() and destination.samefile(source.path)
        ):
            raise PublishError("index destination would replace the source")
        digest = source.identity()
        size = source.stamp.size
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=".resource-index-", dir=destination.parent, delete=False,
            ) as file:
                temporary = Path(file.name)
                file.write(bytes(HEADER.size))
                records_digest = hashlib.sha256()
                previous = None
                count = 0
                for count, item in enumerate(ranges, start=1):
                    if count > MAX_RECORDS:
                        raise IndexFormatError("record count exceeds the index limit")
                    validate_range(item, size, previous)
                    stored = encode_range(item)
                    file.write(stored)
                    records_digest.update(stored)
                    previous = item
                file.write(records_digest.digest())
                file.seek(0)
                file.write(HEADER.pack(MAGIC, SCHEMA, PARSER_VERSION, size, count, digest))
                file.flush()
                os.fsync(file.fileno())
                saved_count, saved_digest = _header(file, size)
                if saved_digest != digest or saved_count != count:
                    raise IndexFormatError("temporary index header changed")
                _validate_records(file, count, size)
            source.check()
            os.replace(temporary, destination)
            temporary = None
        except OSError as error:
            raise PublishError(f"could not publish index to {destination}: {error}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class ResourceIndex:
    _file: BinaryIO
    _source: _Source
    _stamp: _Stamp
    _count: int

    def _check(self) -> None:
        self._source.check()
        if _Stamp.of(os.fstat(self._file.fileno())) != self._stamp:
            raise IndexFormatError("index changed while in use")

    def lookup(self, offset: int, length: int) -> ResourceRange | None:
        """Return a validated hint covering a logical interval, or None for the direct read."""
        end = checked_end(offset, length)
        try:
            self._check()
            lo, hi = 0, self._count
            while lo < hi:
                mid = (lo + hi) // 2
                self._file.seek(HEADER.size + mid * RECORD.size)
                item = decode_range(_read_exact(self._file, RECORD.size))
                validate_range(item, self._source.stamp.size, None)
                if offset < item.logical_offset:
                    hi = mid
                elif offset >= item.logical_end:
                    lo = mid + 1
                else:
                    self._check()
                    return item if end <= item.logical_end else None
        except (IndexFormatError, OSError):
            return None
        return None


@contextmanager
def open_resource(
    root: Path, source_path: Path, sidecar: Path | None = None, *, strict: bool = False,
) -> Iterator[tuple[BinaryIO, ResourceIndex | None]]:
    """Open the source once. Missing or unusable sidecars leave its direct read path available."""
    with _open_source(root, source_path) as source, ExitStack() as stack:
        index = None
        if sidecar is not None:
            try:
                file = stack.enter_context(_open_regular(sidecar))
                stamp = _Stamp.of(os.fstat(file.fileno()))
                count, digest = _header(file, source.stamp.size)
                _validate_records(file, count, source.stamp.size)
                if source.identity() != digest:
                    raise SourceChanged("source SHA-256 does not match the index")
                index = ResourceIndex(file, source, stamp, count)
                index._check()
            except (OSError, IndexFormatError):
                if strict:
                    raise
                stack.close()
                index = None
        yield source.stream, index
