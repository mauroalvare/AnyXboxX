"""Build or inspect a range sidecar from explicitly supplied metadata."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

from . import Codec, Confidence, IndexFormatError, ResourceRange, build_index, open_resource

MAX_LINE_BYTES = 4096
RANGE_FIELDS = {field.name for field in fields(ResourceRange)}


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IndexFormatError(f"duplicate field {key}")
        result[key] = value
    return result


def ranges_from(path: Path):
    with path.open("rb") as stream:
        for number, line in enumerate(iter(lambda: stream.readline(MAX_LINE_BYTES + 1), b""), 1):
            if len(line) > MAX_LINE_BYTES:
                raise IndexFormatError(f"range line {number} exceeds {MAX_LINE_BYTES} bytes")
            try:
                values = json.loads(line.decode("utf-8"), object_pairs_hook=unique_fields)
                if not isinstance(values, dict) or values.keys() != RANGE_FIELDS:
                    raise IndexFormatError("expected exactly the schema 1 range fields")
                values["codec"] = Codec[values["codec"].upper()]
                values["confidence"] = Confidence[values["confidence"].upper()]
                yield ResourceRange(**values)
            except (ValueError, AttributeError, KeyError, RecursionError) as error:
                raise IndexFormatError(f"invalid range at line {number}: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="authorized source directory")
    parser.add_argument("--source", type=Path, required=True, help="source path relative to --root")
    parser.add_argument("--index", type=Path, required=True, help="sidecar path")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="publish a sidecar atomically")
    build.add_argument("ranges", type=Path, help="JSON Lines range metadata in canonical order")
    query = commands.add_parser("query", help="inspect the hint covering a logical interval")
    query.add_argument("offset", type=int)
    query.add_argument("length", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build_index(args.root, args.source, ranges_from(args.ranges), args.index)
        else:
            with open_resource(args.root, args.source, args.index, strict=True) as (_, index):
                item = index.lookup(args.offset, args.length)
                result = None if item is None else {
                    **asdict(item), "codec": item.codec.name.lower(), "confidence": item.confidence.name.lower(),
                }
                print(json.dumps(result, sort_keys=True))
    except (OSError, IndexFormatError) as error:
        parser.exit(1, f"resource index: {error}\n")


if __name__ == "__main__":
    main()
