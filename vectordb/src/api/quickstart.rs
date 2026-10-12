//! What a first run needs so that text goes in and comes back out without configuration: a
//! default embedding model, a table ready for notes, and a startup message that says where
//! everything is.
//!
//! Before this, `vectordb serve` printed nothing at all unless `RUST_LOG` was set, a table
//! could not be created without either a dimension or an `embedding` block naming a model and
//! an endpoint, and a record with `text` was refused on any table without one. Measured on a
//! machine with ollama installed: four pieces of configuration between a downloaded binary and
//! a first search. And the model that machine happened to have, `mxbai-embed-large`, is
//! English-only -- a Chinese query for 索引结构 ranked 晚饭吃了番茄炒蛋 first -- which is why the
//! choice prefers multilingual models and says so when it cannot find one.

use crate::storage::IntelligentIndexedStorage;
use crate::types::table::EmbeddingConfig;

/// Embedding models ollama serves that handle Chinese as well as English, best first.
const MULTILINGUAL: &[&str] = &[
    "bge-m3",
    "qwen3-embedding",
    "embeddinggemma",
    "snowflake-arctic-embed2",
    "granite-embedding:278m",
    "paraphrase-multilingual",
];

/// Embedding models known to be trained on English only: usable, but a Chinese or mixed
/// knowledge base searched with one ranks by surface form rather than meaning.
const ENGLISH_ONLY: &[&str] = &[
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
    "bge-large",
    "bge-base",
    "bge-small",
    "granite-embedding:30m",
];

/// The environment variable the ollama default reads its key from. Unset in practice, which
/// sends no key at all: the default for a model would otherwise be `OPENAI_API_KEY`, and a
/// local model has no business receiving that.
const LOCAL_KEY_ENV: &str = "DXMT_EMBEDDING_API_KEY";

/// Where the server's default model came from, for the startup message.
#[derive(Debug, Clone)]
pub struct DefaultEmbedding {
    pub config: EmbeddingConfig,
    /// "DXMT_EMBEDDING_MODEL", or the ollama address it was found on.
    pub source: String,
}

/// Whether `model` is one of the English-only models above. Tags (`:latest`) are ignored
/// unless the list names one, as it does for the two sizes of `granite-embedding`.
pub fn is_english_only(model: &str) -> bool {
    matches_any(model, ENGLISH_ONLY) && !matches_any(model, MULTILINGUAL)
}

fn matches_any(model: &str, names: &[&str]) -> bool {
    let bare = model.split(':').next().unwrap_or(model);
    names.iter().any(|name| {
        if name.contains(':') {
            model == *name || model.starts_with(&format!("{name}-"))
        } else {
            bare == *name || bare.ends_with(&format!("/{name}"))
        }
    })
}

/// Choose from the models a local ollama has: the first multilingual one in preference order,
/// else the first English-only one, else anything whose name says it embeds.
pub fn choose_model(available: &[String]) -> Option<String> {
    for wanted in MULTILINGUAL {
        if let Some(found) = available.iter().find(|m| matches_any(m, &[wanted])) {
            return Some(found.clone());
        }
    }
    for wanted in ENGLISH_ONLY {
        if let Some(found) = available.iter().find(|m| matches_any(m, &[wanted])) {
            return Some(found.clone());
        }
    }
    available.iter().find(|m| m.contains("embed")).cloned()
}

static RESOLVED: std::sync::OnceLock<Option<DefaultEmbedding>> = std::sync::OnceLock::new();

/// Resolve the default model, give it to `storage`, and remember it for the startup message.
pub async fn configure_default_embedding(storage: &IntelligentIndexedStorage) {
    let resolved = resolve_default_embedding().await;
    storage.set_default_embedding(resolved.as_ref().map(|d| d.config.clone()));
    let _ = RESOLVED.set(resolved);
}

/// What [`configure_default_embedding`] settled on, if it has run.
pub fn resolved_default_embedding() -> Option<&'static DefaultEmbedding> {
    RESOLVED.get().and_then(Option::as_ref)
}

/// The server's default model: `DXMT_EMBEDDING_MODEL` (with `DXMT_EMBEDDING_BASE_URL` and
/// `DXMT_EMBEDDING_API_KEY_ENV`) if set; otherwise an embedding model a local ollama already
/// has, unless `DXMT_EMBEDDING_AUTODETECT=0`; otherwise none.
pub async fn resolve_default_embedding() -> Option<DefaultEmbedding> {
    let var = |name: &str| {
        crate::config::env::var(name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    if let Some(model) = var("DXMT_EMBEDDING_MODEL") {
        return Some(DefaultEmbedding {
            config: EmbeddingConfig {
                model,
                base_url: var("DXMT_EMBEDDING_BASE_URL"),
                field: None,
                api_key_env: var("DXMT_EMBEDDING_API_KEY_ENV"),
                batch_size: None,
            },
            source: "DXMT_EMBEDDING_MODEL".to_string(),
        });
    }
    if var("DXMT_EMBEDDING_AUTODETECT").as_deref() == Some("0") {
        return None;
    }
    detect_ollama().await
}

#[cfg(feature = "server-embedding")]
async fn detect_ollama() -> Option<DefaultEmbedding> {
    let host = crate::config::env::var("OLLAMA_HOST")
        .ok()
        .filter(|h| !h.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1:11434".to_string());
    let base = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
        .ok()?;
    let tags: serde_json::Value = client
        .get(format!("{base}/api/tags"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let available: Vec<String> = tags["models"]
        .as_array()?
        .iter()
        .filter_map(|m| m["name"].as_str().map(str::to_string))
        .collect();
    let model = choose_model(&available)?;
    Some(DefaultEmbedding {
        config: EmbeddingConfig {
            model,
            base_url: Some(format!("{base}/v1")),
            field: None,
            api_key_env: Some(LOCAL_KEY_ENV.to_string()),
            batch_size: None,
        },
        source: format!("ollama at {base}"),
    })
}

#[cfg(not(feature = "server-embedding"))]
async fn detect_ollama() -> Option<DefaultEmbedding> {
    None
}

/// What `serve` prints once it is listening: where everything is, which model text goes
/// through, and the two requests that store and find a note.
pub async fn banner(
    storage: &IntelligentIndexedStorage,
    default_embedding: Option<&DefaultEmbedding>,
    host: &str,
    port: u16,
    swagger: bool,
    dashboard: bool,
    api_key_set: bool,
) -> String {
    let shown = if host == "0.0.0.0" || host == "::" {
        "127.0.0.1"
    } else {
        host
    };
    let url = format!("http://{shown}:{port}");
    let mut out = String::new();
    let mut line = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    line(String::new());
    line(format!(
        "DXMT VectorDB {} is running",
        env!("CARGO_PKG_VERSION")
    ));
    line(format!("  API        {url}/api/v1"));
    if dashboard {
        line(format!("  Console    {url}/dashboard"));
    }
    if swagger {
        line(format!("  API docs   {url}/api/docs"));
    }
    if let Some(dir) = storage.data_dir() {
        line(format!("  Data       {}", dir.display()));
    }
    match default_embedding {
        Some(found) => {
            line(format!(
                "  Embedding  {} ({})",
                found.config.model, found.source
            ));
            if is_english_only(&found.config.model) {
                line(format!(
                    "             {} is trained on English only; Chinese and mixed-language \
                     text will search poorly. `ollama pull bge-m3` and restart for a \
                     multilingual model.",
                    found.config.model
                ));
            }
        }
        None => {
            line("  Embedding  none -- records need vectors, or a table needs a model".to_string());
            line(
                "             To store text directly: `ollama pull bge-m3` and restart, or set \
                 DXMT_EMBEDDING_MODEL (and OPENAI_API_KEY for OpenAI)."
                    .to_string(),
            );
        }
    }
    if host == "0.0.0.0" && !api_key_set {
        line(
            "  Warning    listening on every interface with no DXMT_API_KEY: anyone who can \
             reach this port can read and delete everything. Use --host 127.0.0.1 or set a key."
                .to_string(),
        );
    }
    // The knowledge-base table is made on its first write or search (see
    // `storage::knowledge_base`), so with a model and a name there is always one to point at.
    let table = match (
        crate::storage::knowledge_base::kb_table_name(),
        default_embedding,
    ) {
        (Some(name), Some(_)) => {
            // Listed, not looked up: looking it up would make it.
            let exists = storage
                .list_tables()
                .await
                .map(|tables| tables.iter().any(|t| t.name == name))
                .unwrap_or(false);
            line(format!(
                "  Notes      table '{name}'{}",
                if exists {
                    ""
                } else {
                    " -- made on its first write: text in, text search out"
                }
            ));
            Some(name)
        }
        _ => None,
    };
    if let Some(name) = table {
        line(String::new());
        line("Try it:".to_string());
        line(format!(
            "  curl -X POST {url}/api/v1/tables/{name}/records -H 'content-type: application/json' \\"
        ));
        line(
            "       -d '{\"records\":[{\"id\":\"note-1\",\"text\":\"HNSW 是一种用于向量检索的图索引\"}]}'"
                .to_string(),
        );
        line(format!(
            "  curl -X POST {url}/api/v1/tables/{name}/search -H 'content-type: application/json' \\"
        ));
        line("       -d '{\"text\":\"向量索引是怎么工作的\",\"top_k\":5}'".to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_multilingual_model_is_preferred_over_an_english_one() {
        let available = names(&["mxbai-embed-large:latest", "llama3:8b", "bge-m3:latest"]);
        assert_eq!(choose_model(&available).as_deref(), Some("bge-m3:latest"));
    }

    #[test]
    fn an_english_model_is_taken_when_it_is_all_there_is() {
        let available = names(&["llama3:8b", "mxbai-embed-large:latest"]);
        let chosen = choose_model(&available).unwrap();
        assert_eq!(chosen, "mxbai-embed-large:latest");
        assert!(is_english_only(&chosen));
    }

    #[test]
    fn a_chat_model_is_never_chosen() {
        assert_eq!(choose_model(&names(&["llama3:8b", "qwen2.5:7b"])), None);
    }

    #[test]
    fn the_two_sizes_of_granite_are_told_apart() {
        assert!(!is_english_only("granite-embedding:278m"));
        assert!(is_english_only("granite-embedding:30m"));
        assert!(!is_english_only("bge-m3:latest"));
        assert!(!is_english_only("text-embedding-3-small"));
    }
}
