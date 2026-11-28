"""Versioned sidecars for explicit, authorized byte-range metadata."""

from .format import Codec, Confidence, IndexFormatError, RangeError, ResourceRange
from .index import PublishError, SourceChanged, build_index, open_resource

__all__ = [
    "Codec",
    "Confidence",
    "IndexFormatError",
    "PublishError",
    "RangeError",
    "ResourceRange",
    "SourceChanged",
    "build_index",
    "open_resource",
]
