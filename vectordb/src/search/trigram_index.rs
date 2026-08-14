//! Trigram-based substring search index.
//!
//! Splits text into overlapping 3-char trigrams and builds an inverted index mapping
//! each trigram to the set of record IDs that contain it. A `contains(pattern)` query
//! computes the intersection of posting lists for all trigrams in the pattern — giving
//! O(|trigrams|) lookup with a small false-positive rate verified by a final string
//! match. Inspired by LanceDB's `contains()` filter and PostgreSQL's pg_trgm extension.
//!
//! Limitations: case-sensitive; patterns shorter than 3 chars fall back to a full scan.

use crate::types::RecordId;
use std::collections::{HashMap, HashSet};

pub struct TrigramIndex {
    /// trigram → set of record IDs
    index: HashMap<[u8; 3], HashSet<RecordId>>,
    /// record ID → original text (for false-positive verification)
    texts: HashMap<RecordId, String>,
}

impl TrigramIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            texts: HashMap::new(),
        }
    }

    /// Insert a record with its associated text.
    pub fn insert(&mut self, id: RecordId, text: &str) {
        let trigrams = trigrams(text);
        for tg in trigrams {
            self.index.entry(tg).or_default().insert(id.clone());
        }
        self.texts.insert(id, text.to_string());
    }

    /// Remove a record from the index.
    pub fn delete(&mut self, id: &RecordId) {
        if let Some(text) = self.texts.remove(id) {
            for tg in trigrams(&text) {
                if let Some(set) = self.index.get_mut(&tg) {
                    set.remove(id);
                }
            }
        }
    }

    /// Return all record IDs whose text contains `pattern` as a substring.
    ///
    /// If `pattern` is shorter than 3 chars, falls back to a full scan.
    pub fn contains(&self, pattern: &str) -> Vec<RecordId> {
        let pat_trigrams = trigrams(pattern);

        if pat_trigrams.is_empty() {
            // Short pattern: brute-force scan
            return self.texts.iter()
                .filter(|(_, text)| text.contains(pattern))
                .map(|(id, _)| id.clone())
                .collect();
        }

        // Start with the posting list of the first trigram
        let mut candidates: HashSet<RecordId> = self.index
            .get(&pat_trigrams[0])
            .cloned()
            .unwrap_or_default();

        // Intersect with each subsequent trigram's list
        for tg in &pat_trigrams[1..] {
            let posting = self.index.get(tg).cloned().unwrap_or_default();
            candidates = candidates.intersection(&posting).cloned().collect();
            if candidates.is_empty() {
                return vec![];
            }
        }

        // Verify (eliminate false positives)
        candidates.into_iter()
            .filter(|id| {
                self.texts.get(id).is_some_and(|t| t.contains(pattern))
            })
            .collect()
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let texts_bytes: usize = self.texts.values().map(|t| t.len() + 32).sum();
        let index_bytes: usize = self.index.values().map(|s| s.len() * 32 + 48).sum();
        texts_bytes + index_bytes + self.index.len() * 8
    }
}

impl Default for TrigramIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute all 3-character byte trigrams for a string.
fn trigrams(s: &str) -> Vec<[u8; 3]> {
    let bytes = s.as_bytes();
    if bytes.len() < 3 {
        return vec![];
    }
    bytes.windows(3)
        .map(|w| [w[0], w[1], w[2]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigram_basic_contains() {
        let mut idx = TrigramIndex::new();
        idx.insert("doc1".into(), "hello world");
        idx.insert("doc2".into(), "foo bar baz");
        idx.insert("doc3".into(), "world peace");

        let mut results = idx.contains("world");
        results.sort();
        let mut expected: Vec<RecordId> = vec!["doc1".into(), "doc3".into()];
        expected.sort();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_trigram_no_match() {
        let mut idx = TrigramIndex::new();
        idx.insert("doc1".into(), "hello world");
        let results = idx.contains("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_trigram_short_pattern() {
        let mut idx = TrigramIndex::new();
        idx.insert("doc1".into(), "hello");
        idx.insert("doc2".into(), "world");

        let results = idx.contains("he");
        let expected: Vec<RecordId> = vec!["doc1".into()];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_trigram_delete() {
        let mut idx = TrigramIndex::new();
        idx.insert("doc1".into(), "machine learning");
        idx.insert("doc2".into(), "deep learning");

        idx.delete(&"doc1".into());

        let results = idx.contains("learning");
        let expected: Vec<RecordId> = vec!["doc2".into()];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_trigram_exact_length() {
        let mut idx = TrigramIndex::new();
        idx.insert("a".into(), "abc");
        let results = idx.contains("abc");
        let expected: Vec<RecordId> = vec!["a".into()];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_trigram_memory_usage() {
        let mut idx = TrigramIndex::new();
        for i in 0..100 {
            idx.insert(format!("doc{i}").into(), &format!("text number {i} with some words"));
        }
        assert!(idx.memory_usage() > 0);
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn test_trigram_partial_overlap() {
        let mut idx = TrigramIndex::new();
        idx.insert("a".into(), "pattern matching in text");
        idx.insert("b".into(), "pattern recognition");
        idx.insert("c".into(), "no match here");

        let mut results = idx.contains("pattern");
        results.sort();
        let mut expected: Vec<RecordId> = vec!["a".into(), "b".into()];
        expected.sort();
        assert_eq!(results, expected);
    }
}
