use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use victory_hle::known_pipelines::WarmCacheMetadata;

const VERSION: &str = "v1";
const COMPLETE: &str = "\tend";

pub(super) fn fingerprint(metadata: WarmCacheMetadata) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(b"victory warm compiler context\0");
    hash.update(&metadata.translation);
    hash.update(&[u8::from(metadata.software)]);
    for part in metadata.adapter {
        hash.update(&part.to_le_bytes());
    }
    hash.update(&metadata.driver.to_le_bytes());
    hash.finalize().to_hex()[..32].to_owned()
}

pub(super) fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))
}

pub(super) struct Journal<'a> {
    pub(super) keys: HashSet<String>,
    fingerprint: Option<&'a str>,
    file: Mutex<File>,
}

impl<'a> Journal<'a> {
    pub(super) fn open(path: &Path, fingerprint: Option<&'a str>) -> Result<Self> {
        let keys = match fingerprint {
            Some(fingerprint) => fs::read(path)
                .unwrap_or_default()
                .split_inclusive(|&b| b == b'\n')
                .filter_map(|line| key_of(line, fingerprint).map(str::to_owned))
                .collect(),
            None => HashSet::new(),
        };
        let file = Mutex::new(open_append(path)?);
        Ok(Self {
            keys,
            fingerprint,
            file,
        })
    }

    pub(super) fn append(&self, key: &str, outcome: &str) -> Result<()> {
        if let Some(fingerprint) = self.fingerprint {
            let outcome = outcome.replace(['\r', '\n'], " ");
            // Start on a new line even when a previous writer left a torn tail.
            // The terminator lets a reader distinguish that tail from a record.
            let record = format!("\n{VERSION} {fingerprint} {key} {outcome}{COMPLETE}\n");
            super::lock(&self.file).write_all(record.as_bytes())?;
        }
        Ok(())
    }
}

fn key_of<'a>(line: &'a [u8], fingerprint: &str) -> Option<&'a str> {
    let line = line.strip_suffix(b"\n")?;
    let fields = std::str::from_utf8(line)
        .ok()?
        .trim_end_matches('\r')
        .strip_suffix(COMPLETE)?;
    let mut parts = fields.splitn(4, ' ');
    let (version, of, key, outcome) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    (version == VERSION
        && of == fingerprint
        && key.len() == 32
        && key.bytes().all(|b| b.is_ascii_hexdigit())
        && !outcome.trim().is_empty())
    .then_some(key)
}
