//! On-disk envelope for serialized indexes.
//!
//! Index files outlive the binary that wrote them: someone upgrades, restarts, and the process
//! reads a file written by a different version of these structs. Bare `bincode::serialize`
//! gives that case no chance — a changed field order deserializes into plausible nonsense, or
//! fails with a message about an invalid length that says nothing about the real cause.
//!
//! So every index file starts with a header naming what it is and what version wrote it, and
//! loading refuses anything it does not recognise. A refused load is recoverable: the caller
//! rebuilds the index from the WAL. A silently mis-parsed one is not.

use serde::{de::DeserializeOwned, Serialize};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"DXMTIDX\x00";

/// Bump when the envelope itself changes shape. Individual index layouts are versioned by
/// `kind`, which is compared verbatim — rename the kind when a struct changes incompatibly.
const FORMAT_VERSION: u32 = 1;

/// Serialize `value` to `path`, tagged with `kind`.
///
/// Writes to a sibling temporary file and renames it into place: a crash midway through leaves
/// the previous index intact rather than a truncated file that loads as an empty index and
/// quietly loses every vector.
pub fn write_index<T: Serialize>(path: &Path, kind: &str, value: &T) -> crate::Result<()> {
    let payload =
        bincode::serialize(value).map_err(|e| crate::Error::Serialization(e.to_string()))?;

    let kind_bytes = kind.as_bytes();
    let mut buf = Vec::with_capacity(payload.len() + kind_bytes.len() + 16);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&(kind_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(kind_bytes);
    buf.extend_from_slice(&payload);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&buf)?;
        file.sync_data()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a value written by [`write_index`], requiring the same `kind`.
pub fn read_index<T: DeserializeOwned>(path: &Path, kind: &str) -> crate::Result<T> {
    let mut file = std::fs::File::open(path)?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).map_err(|_| {
        crate::Error::Serialization(format!("{}: too short to be an index file", path.display()))
    })?;
    if &magic != MAGIC {
        return Err(crate::Error::Serialization(format!(
            "{}: not a DXMT index file",
            path.display()
        )));
    }

    let mut version = [0u8; 4];
    file.read_exact(&mut version)?;
    let version = u32::from_le_bytes(version);
    if version != FORMAT_VERSION {
        return Err(crate::Error::Serialization(format!(
            "{}: index format version {}, this build writes {}; rebuild the index",
            path.display(),
            version,
            FORMAT_VERSION
        )));
    }

    let mut kind_len = [0u8; 4];
    file.read_exact(&mut kind_len)?;
    let mut found_kind = vec![0u8; u32::from_le_bytes(kind_len) as usize];
    file.read_exact(&mut found_kind)?;
    let found_kind = String::from_utf8_lossy(&found_kind).into_owned();
    if found_kind != kind {
        return Err(crate::Error::Serialization(format!(
            "{}: holds a '{}' index, expected '{}'",
            path.display(),
            found_kind,
            kind
        )));
    }

    let mut payload = Vec::new();
    file.read_to_end(&mut payload)?;
    bincode::deserialize(&payload).map_err(|e| crate::Error::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trips() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("idx.bin");
        write_index(&path, "test", &vec![1u32, 2, 3]).unwrap();
        let back: Vec<u32> = read_index(&path, "test").unwrap();
        assert_eq!(back, vec![1, 2, 3]);
    }

    #[test]
    fn rejects_a_different_kind() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("idx.bin");
        write_index(&path, "hnsw", &vec![1u32]).unwrap();

        // The whole point: loading an hnsw file as hnsw-pq must fail loudly rather than
        // deserialize into a structurally valid but wrong index.
        let err = read_index::<Vec<u32>>(&path, "hnsw-pq").unwrap_err().to_string();
        assert!(err.contains("holds a 'hnsw' index"), "unhelpful error: {}", err);
    }

    #[test]
    fn rejects_a_foreign_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("idx.bin");
        std::fs::write(&path, b"this is not an index at all").unwrap();

        let err = read_index::<Vec<u32>>(&path, "hnsw").unwrap_err().to_string();
        assert!(err.contains("not a DXMT index file"), "unhelpful error: {}", err);
    }

    #[test]
    fn leaves_the_previous_file_intact_when_the_write_target_is_reused() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("idx.bin");
        write_index(&path, "test", &vec![1u32, 2, 3]).unwrap();
        write_index(&path, "test", &vec![9u32]).unwrap();

        let back: Vec<u32> = read_index(&path, "test").unwrap();
        assert_eq!(back, vec![9], "rewrite should replace, not append or corrupt");
        assert!(!path.with_extension("tmp").exists(), "temp file should be renamed away");
    }
}
