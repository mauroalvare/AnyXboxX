"""Resource-index schema 1. All integers on disk are little endian."""

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct

MAGIC = b"AXXINDEX"
SCHEMA = 1
PARSER_VERSION = 1
HEADER = Struct("<8sHHQQ32s")
RECORD = Struct("<QQQQQBBH")
DIGEST_BYTES = 32
MAX_RECORDS = 1 << 20
MAX_EXPANSION = 64 * 1024 * 1024
U64_MAX = (1 << 64) - 1


class IndexFormatError(ValueError):
    """An unusable index or invalid input metadata."""


class RangeError(IndexFormatError):
    """An invalid or ambiguous byte range."""


class Codec(IntEnum):
    RAW = 0
    ZLIB = 1
    BCPACK = 2


class Confidence(IntEnum):
    DECLARED = 1
    CONFIRMED = 2


@dataclass(frozen=True, slots=True)
class ResourceRange:
    physical_offset: int
    physical_length: int
    logical_offset: int
    logical_length: int
    expansion_limit: int
    codec: Codec
    confidence: Confidence

    @property
    def logical_end(self) -> int:
        return self.logical_offset + self.logical_length


def checked_end(offset: int, length: int) -> int:
    if type(offset) is not int or type(length) is not int:
        raise RangeError("offset and length must be integers")
    if offset < 0 or length <= 0 or offset > U64_MAX - length:
        raise RangeError("range is empty or exceeds unsigned 64-bit bounds")
    return offset + length


def validate_range(item: ResourceRange, source_size: int, previous: ResourceRange | None) -> None:
    physical_end = checked_end(item.physical_offset, item.physical_length)
    checked_end(item.logical_offset, item.logical_length)
    if physical_end > source_size:
        raise RangeError("physical range exceeds the source length")
    if type(item.expansion_limit) is not int or not item.logical_length <= item.expansion_limit <= MAX_EXPANSION:
        raise RangeError("logical length exceeds the declared or supported expansion limit")
    if not isinstance(item.codec, Codec):
        raise RangeError("unknown codec")
    if not isinstance(item.confidence, Confidence):
        raise RangeError("unknown or ambiguous range confidence")
    if item.codec is Codec.RAW and item.logical_length != item.physical_length:
        raise RangeError("raw physical and logical lengths differ")
    if previous is not None and (
        item.physical_offset < previous.physical_offset + previous.physical_length
        or item.logical_offset < previous.logical_end
    ):
        raise RangeError("ranges must be monotonic and non-overlapping in both address spaces")


def encode_range(item: ResourceRange) -> bytes:
    return RECORD.pack(
        item.physical_offset, item.physical_length, item.logical_offset,
        item.logical_length, item.expansion_limit, item.codec, item.confidence, 0,
    )


def decode_range(data: bytes) -> ResourceRange:
    physical, stored, logical, expanded, limit, codec, confidence, reserved = RECORD.unpack(data)
    if reserved:
        raise RangeError("nonzero reserved range bits")
    try:
        return ResourceRange(physical, stored, logical, expanded, limit, Codec(codec), Confidence(confidence))
    except ValueError as error:
        raise RangeError("unknown codec or ambiguous range confidence") from error
