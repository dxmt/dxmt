//! Hashing for id lookup tables, shared by the graph and the record store.
//!
//! Both keep ids once in a byte arena and find a row or node through a
//! `HashMap<u64, u32>` keyed by a 64-bit hash of the id. Two costs showed up in a profile of
//! the 1M benchmark under load, together 8.4% of the server's CPU:
//!
//! - **The key was hashed twice.** `id_hash` produces a well-distributed 64-bit value and
//!   `HashMap`'s default `SipHash` then hashed that again. [`IdHashMap`] passes the key
//!   through instead, which is what a map keyed by a hash wants.
//! - **Every comparison validated UTF-8.** Confirming a hash hit means comparing the stored
//!   id with the queried one, and reading the stored bytes back as `&str` runs a full
//!   validation pass first. Equality does not need it; [`bytes_eq`] compares the bytes.
//!
//! Neither changes what is stored or which row is found, so both are invisible outside this
//! module beyond being faster.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// FNV-1a over the id's bytes.
///
/// Spelled out rather than taken from `DefaultHasher` because it is written into index files
/// and must not change when the standard library's hasher does.
pub fn id_hash(id: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Whether the arena bytes at `stored` are the same id as `id`.
///
/// The arena is only ever filled from `&str`, so the bytes are valid UTF-8 and comparing
/// them is exactly comparing the strings.
#[inline]
pub fn bytes_eq(stored: &[u8], id: &str) -> bool {
    stored == id.as_bytes()
}

/// Passes a `u64` key through unchanged.
///
/// Safe here because the key is already [`id_hash`]'s output rather than user input: it is
/// avalanched, so the map's own bucket selection has good bits to work with. Do not use it
/// for a map whose keys a caller chooses.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only the u64 path is used; this keeps the trait honest if that ever changes.
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ (*b as u64);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

/// A `HashMap` keyed by an [`id_hash`] value.
pub type IdHashMap<V> = HashMap<u64, V, BuildHasherDefault<IdentityHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_the_documented_fnv_1a_and_does_not_drift() {
        // Written into index files, so these are fixed points, not just any hash.
        assert_eq!(id_hash(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(id_hash("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(id_hash("foobar"), 0x85944171f73967e8);
        assert_ne!(id_hash("rec-1"), id_hash("rec-2"));
    }

    #[test]
    fn bytes_eq_matches_string_equality_including_the_empty_and_non_ascii_cases() {
        for id in ["", "a", "rec-1", "键", "a\u{0}b"] {
            assert!(bytes_eq(id.as_bytes(), id), "{id:?}");
            assert!(!bytes_eq(id.as_bytes(), "different"), "{id:?}");
        }
        assert!(!bytes_eq(b"ab", "abc"));
        assert!(!bytes_eq(b"abc", "ab"));
    }

    #[test]
    fn the_identity_map_stores_and_finds_what_a_default_map_would() {
        let mut fast: IdHashMap<u32> = IdHashMap::default();
        let mut plain: HashMap<u64, u32> = HashMap::new();
        for i in 0..1000u32 {
            let k = id_hash(&format!("rec-{i}"));
            fast.insert(k, i);
            plain.insert(k, i);
        }
        assert_eq!(fast.len(), plain.len());
        for (k, v) in &plain {
            assert_eq!(fast.get(k), Some(v));
        }
        for i in 0..1000u32 {
            let k = id_hash(&format!("rec-{i}"));
            assert_eq!(fast.remove(&k), Some(i));
        }
        assert!(fast.is_empty());
        assert_eq!(fast.get(&id_hash("missing")), None);
    }
}
