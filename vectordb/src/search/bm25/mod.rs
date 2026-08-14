use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::types::RecordId;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Document {
    pub id: RecordId,
    pub text: String,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingListEntry {
    pub doc_id: RecordId,
    pub term_frequency: u32,
    pub positions: Vec<u32>,
}

#[derive(Clone)]
pub struct BM25Index {
    inverted_index: Arc<RwLock<HashMap<String, Vec<PostingListEntry>>>>,
    doc_lengths: Arc<RwLock<HashMap<RecordId, u32>>>,
    doc_frequencies: Arc<RwLock<HashMap<String, u32>>>,
    total_docs: Arc<RwLock<u64>>,
    avg_doc_length: Arc<RwLock<f64>>,
    k1: f32,
    b: f32,
}

impl BM25Index {
    pub fn new() -> Self {
        Self::with_params(K1, B)
    }

    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            inverted_index: Arc::new(RwLock::new(HashMap::new())),
            doc_lengths: Arc::new(RwLock::new(HashMap::new())),
            doc_frequencies: Arc::new(RwLock::new(HashMap::new())),
            total_docs: Arc::new(RwLock::new(0)),
            avg_doc_length: Arc::new(RwLock::new(0.0)),
            k1,
            b,
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    pub fn insert_document(&self, doc: BM25Document) {
        let tokens = if doc.tokens.is_empty() {
            self.tokenize(&doc.text)
        } else {
            doc.tokens
        };

        let doc_length = tokens.len() as u32;

        let mut term_counts: HashMap<String, u32> = HashMap::new();
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();

        for (position, token) in tokens.iter().enumerate() {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
            term_positions.entry(token.clone())
                .or_default()
                .push(position as u32);
        }

        let mut inverted_index = self.inverted_index.write();
        let mut doc_lengths = self.doc_lengths.write();
        let mut doc_frequencies = self.doc_frequencies.write();
        let mut total_docs = self.total_docs.write();
        let mut avg_doc_length = self.avg_doc_length.write();

        doc_lengths.insert(doc.id.clone(), doc_length);

        for (term, tf) in term_counts {
            let posting_entry = PostingListEntry {
                doc_id: doc.id.clone(),
                term_frequency: tf,
                positions: term_positions.get(&term).cloned().unwrap_or_default(),
            };

            inverted_index.entry(term.clone())
                .or_default()
                .push(posting_entry);

            *doc_frequencies.entry(term).or_insert(0) += 1;
        }

        let old_total = *total_docs;
        *total_docs += 1;
        *avg_doc_length = (*avg_doc_length * old_total as f64 + doc_length as f64) / *total_docs as f64;
    }

    pub fn delete_document(&self, doc_id: &RecordId) -> bool {
        let mut inverted_index = self.inverted_index.write();
        let mut doc_lengths = self.doc_lengths.write();
        let mut doc_frequencies = self.doc_frequencies.write();
        let mut total_docs = self.total_docs.write();
        let mut avg_doc_length = self.avg_doc_length.write();

        if let Some(doc_length) = doc_lengths.remove(doc_id) {
            let mut terms_to_update = Vec::new();

            for (term, postings) in inverted_index.iter() {
                if postings.iter().any(|p| &p.doc_id == doc_id) {
                    terms_to_update.push(term.clone());
                }
            }

            for term in terms_to_update {
                if let Some(postings) = inverted_index.get_mut(&term) {
                    postings.retain(|p| &p.doc_id != doc_id);

                    if postings.is_empty() {
                        inverted_index.remove(&term);
                        doc_frequencies.remove(&term);
                    } else {
                        *doc_frequencies.get_mut(&term).unwrap() -= 1;
                    }
                }
            }

            if *total_docs > 1 {
                *avg_doc_length = (*avg_doc_length * *total_docs as f64 - doc_length as f64)
                    / (*total_docs - 1) as f64;
                *total_docs -= 1;
            } else {
                *total_docs = 0;
                *avg_doc_length = 0.0;
            }

            true
        } else {
            false
        }
    }

    pub fn update_document(&self, doc: BM25Document) {
        self.delete_document(&doc.id);
        self.insert_document(doc);
    }

    fn calculate_bm25_score(
        &self,
        tf: u32,
        doc_len: u32,
        df: u32,
        total_docs: u64,
        avgdl: f64,
    ) -> f32 {
        let df = df.max(1) as f64;
        let total_docs = total_docs.max(1) as f64;

        let idf = (((total_docs - df + 0.5) / (df + 0.5)) + 1.0).ln();

        let avgdl = if avgdl > 0.0 { avgdl } else { doc_len as f64 };

        let tf = tf as f64;
        let doc_len = doc_len as f64;
        let tf_component = (tf * (self.k1 as f64 + 1.0))
            / (tf + self.k1 as f64 * (1.0 - self.b as f64 + self.b as f64 * (doc_len / avgdl)));

        (idf * tf_component) as f32
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<(RecordId, f32)> {
        let query_terms = self.tokenize(query);

        if query_terms.is_empty() {
            return Vec::new();
        }

        let inverted_index = self.inverted_index.read();
        let doc_lengths = self.doc_lengths.read();
        let doc_frequencies = self.doc_frequencies.read();
        let total_docs = *self.total_docs.read();
        let avg_doc_length = *self.avg_doc_length.read();

        let mut doc_scores: HashMap<RecordId, f32> = HashMap::new();

        for term in query_terms {
            let doc_frequency = doc_frequencies.get(&term).copied().unwrap_or(0);

            if doc_frequency == 0 {
                continue;
            }

            if let Some(postings) = inverted_index.get(&term) {
                for posting in postings {
                    let doc_length = doc_lengths.get(&posting.doc_id).copied().unwrap_or(0);

                    let score = self.calculate_bm25_score(
                        posting.term_frequency,
                        doc_length,
                        doc_frequency,
                        total_docs,
                        avg_doc_length,
                    );

                    *doc_scores.entry(posting.doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(RecordId, f32)> = doc_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }

    pub fn phrase_search(&self, phrase: &str, limit: usize) -> Vec<(RecordId, f32)> {
        let terms = self.tokenize(phrase);

        if terms.is_empty() {
            return Vec::new();
        }

        if terms.len() == 1 {
            return self.search(&terms[0], limit);
        }

        let inverted_index = self.inverted_index.read();
        let doc_lengths = self.doc_lengths.read();

        let mut candidate_docs: HashSet<RecordId> = HashSet::new();

        if let Some(first_postings) = inverted_index.get(&terms[0]) {
            for posting in first_postings {
                candidate_docs.insert(posting.doc_id.clone());
            }
        }

        for term in &terms[1..] {
            let mut new_candidates = HashSet::new();

            if let Some(postings) = inverted_index.get(term) {
                for posting in postings {
                    if candidate_docs.contains(&posting.doc_id) {
                        new_candidates.insert(posting.doc_id.clone());
                    }
                }
            }

            candidate_docs = new_candidates;

            if candidate_docs.is_empty() {
                return Vec::new();
            }
        }

        let mut doc_scores: Vec<(RecordId, f32)> = Vec::new();

        for doc_id in candidate_docs {
            let mut term_positions: Vec<Vec<u32>> = Vec::new();

            for term in &terms {
                if let Some(postings) = inverted_index.get(term) {
                    if let Some(posting) = postings.iter().find(|p| p.doc_id == doc_id) {
                        term_positions.push(posting.positions.clone());
                    } else {
                        term_positions.push(Vec::new());
                    }
                }
            }

            if term_positions.iter().any(|p| p.is_empty()) {
                continue;
            }

            let mut phrase_found = false;
            for &start_pos in &term_positions[0] {
                let mut all_match = true;

                for (i, positions) in term_positions[1..].iter().enumerate() {
                    let expected_pos = start_pos + (i + 1) as u32;
                    if !positions.contains(&expected_pos) {
                        all_match = false;
                        break;
                    }
                }

                if all_match {
                    phrase_found = true;
                    break;
                }
            }

            if phrase_found {
                let score = self.calculate_phrase_score(&doc_id, &terms, &doc_lengths);
                doc_scores.push((doc_id, score));
            }
        }

        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        doc_scores.truncate(limit);

        doc_scores
    }

    fn calculate_phrase_score(
        &self,
        doc_id: &RecordId,
        terms: &[String],
        doc_lengths: &HashMap<RecordId, u32>,
    ) -> f32 {
        let doc_length = doc_lengths.get(doc_id).copied().unwrap_or(1) as f32;
        let phrase_length = terms.len() as f32;

        let base_score = phrase_length / doc_length.sqrt();
        let boost = 2.0;

        base_score * boost
    }

    pub fn stats(&self) -> BM25Stats {
        BM25Stats {
            total_docs: *self.total_docs.read(),
            avg_doc_length: *self.avg_doc_length.read(),
            unique_terms: self.inverted_index.read().len(),
            total_postings: self.inverted_index.read()
                .values()
                .map(|p| p.len())
                .sum(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Stats {
    pub total_docs: u64,
    pub avg_doc_length: f64,
    pub unique_terms: usize,
    pub total_postings: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic_operations() {
        let index = BM25Index::new();

        let doc1 = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "The quick brown fox jumps over the lazy dog".to_string(),
            tokens: vec![],
        };

        let doc2 = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "The lazy dog sleeps under the tree".to_string(),
            tokens: vec![],
        };

        let doc3 = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "A quick brown fox runs through the forest".to_string(),
            tokens: vec![],
        };

        index.insert_document(doc1.clone());
        index.insert_document(doc2.clone());
        index.insert_document(doc3.clone());

        let results = index.search("quick fox", 10);
        assert_eq!(results.len(), 2);

        assert!(results[0].0 == doc1.id || results[0].0 == doc3.id);
    }

    #[test]
    fn test_phrase_search() {
        let index = BM25Index::new();

        let doc1 = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "The quick brown fox jumps over the lazy dog".to_string(),
            tokens: vec![],
        };

        let doc2 = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "A quick brown dog runs fast".to_string(),
            tokens: vec![],
        };

        index.insert_document(doc1.clone());
        index.insert_document(doc2.clone());

        let results = index.phrase_search("quick brown fox", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc1.id);

        let results = index.phrase_search("quick brown dog", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc2.id);
    }

    #[test]
    fn test_delete_document() {
        let index = BM25Index::new();

        let doc = BM25Document {
            id: uuid::Uuid::new_v4().to_string(),
            text: "Test document for deletion".to_string(),
            tokens: vec![],
        };

        index.insert_document(doc.clone());

        let results = index.search("test document", 10);
        assert_eq!(results.len(), 1);

        assert!(index.delete_document(&doc.id));

        let results = index.search("test document", 10);
        assert_eq!(results.len(), 0);
    }
}