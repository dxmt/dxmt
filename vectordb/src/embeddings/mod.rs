//! Embeddings — the one file a request reaches.
//!
//! [`service`] is it: a table that names a model gets its text turned into a vector there, on
//! insert and on search. It is behind `server-embedding`, a default feature.
//!
//! The four overlapping provider abstractions that used to sit beside it (`EmbeddingFunction`,
//! `EmbeddingProvider`, `EmbeddingClient`, `AutoEmbedding` — 2,400 lines behind a non-default
//! `embeddings` feature, each opening with `#![allow(dead_code)]`) were taken out of the crate
//! on 2026-09-27. Nothing called them.

/// The rerank provider client, behind the same feature: it is the same HTTP client and the
/// same key-in-the-environment contract.
#[cfg(feature = "server-embedding")]
pub mod rerank;
#[cfg(feature = "server-embedding")]
pub mod service;
