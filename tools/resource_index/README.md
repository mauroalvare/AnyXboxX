# Resource range index

This tool writes a sidecar for explicit byte-range metadata from an authorized source. It
reads the source to compute SHA-256 and treats every range as a hint. The caller keeps the
source open for its normal reads and checks a hint against the request's offset, size and codec.
An absent, stale or invalid sidecar yields the same source stream with no index. The command-line
query reports incompatibilities so a producer can diagnose them.

Run from the repository root. The example assumes a fixture directory containing `data.bin`
and a JSON Lines file named `ranges.jsonl` in the current directory.

```powershell
python -m tools.resource_index --root fixtures --source data.bin --index fixture.axxi build ranges.jsonl
python -m tools.resource_index --root fixtures --source data.bin --index fixture.axxi query 0 2
```

Each input line contains one object with the seven fields shown below. List ranges in ascending
order in both address spaces. Intervals are half-open and may be adjacent. Unknown confidence,
overlap and out-of-order records make the build fail. Supply only ranges with a resolved meaning.

```json
{"physical_offset":0,"physical_length":8,"logical_offset":0,"logical_length":8,"expansion_limit":8,"codec":"raw","confidence":"confirmed"}
```

## Schema 1, parser 1

All fields on disk are little endian. Offsets and lengths are unsigned 64-bit values. The file
contains one header, the exact number of fixed-size records stated by that header, and a SHA-256
of the concatenated record bytes. A reader checks the versions, length, bounds, order and record
digest before using the table. The source length and SHA-256 identify the source independently
of its path and timestamps.

| Header field | Encoding |
| --- | --- |
| Magic | ASCII `AXXINDEX`, 8 bytes |
| Schema and parser versions | Two `u16` values, both `1` |
| Source length and record count | Two `u64` values |
| Source identity | 32 raw SHA-256 bytes |

| Record field | Encoding |
| --- | --- |
| Physical offset and length | Two `u64` values bounded by the source length |
| Logical offset and length | Two `u64` values with checked end |
| Expansion limit in bytes | `u64`, at least the logical length and at most 64 MiB |
| Codec | `u8`, raw `0`, zlib `1`, BCPack `2` |
| Confidence | `u8`, declared `1`, confirmed `2` |
| Reserved | `u16` zero |

Raw ranges have equal physical and logical lengths. The codec and confidence describe supplied
metadata. SHA-256 detects changes to bytes and does not grant authority to the range declarations.
Schema 1 carries up to 1,048,576 records. CLI input lines are limited to 4,096 bytes. Build and
validation stream records with fixed buffers. A lookup binary-searches the verified file and
checks the source and index handles for changes. Use one context per reader so seeks stay local
to the request stream. The source hash is computed once per indexed open.

`open_resource(root, relative_source, sidecar)` yields `(source, index)`. When `index` is
available, `index.lookup(logical_offset, length)` returns a covering hint or `None`. Read using
the existing path when no hint matches the request. `strict=True` reports an unusable sidecar.

The source must resolve to a regular file under the supplied root. The builder writes a
temporary file next to the destination, flushes and validates it, and then publishes it with a
single atomic replace. A failed build leaves the previous sidecar available. Keep the source
and the output directory under the caller's control during the build. Publish generated indexes
only for fixtures or data authorized for sharing.
