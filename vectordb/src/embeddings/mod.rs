//! Embeddings — one reachable file and one directory of code that is not.
//!
//! [`service`] is what a request reaches: a table that names a model gets its text turned into
//! a vector there, on insert and on search. It is behind `server-embedding`, a default feature.
//!
//! Everything in [`legacy`] is behind `embeddings`, which is not a default feature and which
//! nothing in the server calls. The two names are close enough to be worth spelling out once:
//! `server-embedding` is the product feature, `embeddings` compiles the unreachable providers.

#[cfg(feature = "server-embedding")]
pub mod service;

#[cfg(feature = "embeddings")]
mod legacy;
#[cfg(feature = "embeddings")]
pub use legacy::*;
