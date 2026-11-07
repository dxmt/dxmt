//! Memory the server derives with a language model: facts out of a conversation, and a profile
//! of each person out of their facts.
//!
//! Kept turns answer "what was said"; they answer "what pets does Nate have" only when one turn
//! happens to say all of it, and "when did Caroline go to the workshop" only when the answer
//! model works out that "last Friday", said on 27 June, means the Friday before. Measured on
//! LoCoMo (five conversations, 100 questions, one 35B model answering and judging every
//! engine, `eval/memory/locomo_answer.py`): an agent answering from kept turns alone scored
//! 0.65; with the facts this module extracts beside them, 0.71; with the profiles too, 0.75.
//! Honcho's own answering agent scored 0.73 on the same questions and OpenViking 0.46, and
//! both spent several times more model calls deriving what they keep.
//!
//! - `extract`: one call per conversation (or session) turns it into self-contained facts,
//!   each naming its person and carrying an absolute date -- "last weekend" becomes "the
//!   weekend before 17 July 2023" -- kept as memories of kind `fact` with the caller's tags.
//! - `profile`: the facts about one person, condensed into one memory of kind `profile`,
//!   grouped by topic. Remade in place: its id is fixed by the person and the tags.
//!
//! Off unless `DXMT_MEMORY_LLM_MODEL` names a chat model. It speaks the OpenAI chat API at
//! `DXMT_MEMORY_LLM_BASE_URL` (default: the embedding provider's), with the key in the variable
//! `DXMT_MEMORY_LLM_API_KEY_ENV` names (default `OPENAI_API_KEY`), under the same rules that
//! decide where an embedding key may be sent (`embeddings::policy`).

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// What `extract` takes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ExtractRequest {
    /// The conversation, one line per turn: "Caroline: I went to a counseling workshop last
    /// Friday". Any text works; a dialogue is what the prompt is written for.
    pub text: String,
    /// When it was held, in words or RFC 3339: relative times in it are resolved against this.
    /// Without it, they are left as said.
    #[serde(default)]
    pub when: Option<String>,
    /// Put on every fact, with the caller's own tags: a user, a conversation, a workspace.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Which agent or conversation it came from; kept on every fact.
    #[serde(default)]
    pub source: Option<String>,
    /// Ids are `{id_prefix}:F{n}`, so extracting the same conversation again replaces its facts
    /// instead of adding a second set. Without it, each fact gets a new id.
    #[serde(default)]
    pub id_prefix: Option<String>,
    /// The tenant this request acts for, set by the server from the caller's key -- never from
    /// the body.
    #[serde(skip)]
    pub scope: Option<String>,
}

/// What `profile` takes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ProfileRequest {
    /// Whose profile: the name the facts use.
    pub name: String,
    /// Only facts with any of these tags, and the profile carries them.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// The tenant this request acts for, set by the server from the caller's key -- never from
    /// the body.
    #[serde(skip)]
    pub scope: Option<String>,
}

/// The model the server derives memory with, from the environment; `None` when it has none.
#[derive(Debug, Clone)]
pub struct DeriveModel {
    pub model: String,
    pub base_url: String,
    pub api_key_env: String,
}

impl DeriveModel {
    pub fn from_env() -> Option<Self> {
        let var = |k: &str| {
            crate::config::env::var(k)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let model = var("DXMT_MEMORY_LLM_MODEL")?;
        let base_url = var("DXMT_MEMORY_LLM_BASE_URL")
            .or_else(|| var("DXMT_EMBEDDING_BASE_URL"))
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Some(Self {
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key_env: var("DXMT_MEMORY_LLM_API_KEY_ENV")
                .unwrap_or_else(|| "OPENAI_API_KEY".to_string()),
        })
    }

    pub fn required() -> Result<Self> {
        Self::from_env().ok_or_else(|| {
            Error::InvalidConfig(
                "deriving memory needs a chat model: set DXMT_MEMORY_LLM_MODEL (and \
                 DXMT_MEMORY_LLM_BASE_URL for a provider other than the embedding one)"
                    .to_string(),
            )
        })
    }

    /// One answer from the model, at temperature 0, with any `<think>` block removed.
    ///
    /// `max_tokens` is always sent: a model that falls into repeating itself otherwise writes
    /// to the end of its context, which on a local server held the model for most of an hour.
    #[cfg(not(feature = "server-embedding"))]
    pub async fn complete(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
        Err(Error::InvalidConfig(
            "deriving memory needs a build with the `server-embedding` feature, which carries \
             the HTTP client for the memory model"
                .to_string(),
        ))
    }

    /// One answer from the model, at temperature 0, with any `<think>` block removed.
    ///
    /// `max_tokens` is always sent: a model that falls into repeating itself otherwise writes
    /// to the end of its context, which on a local server held the model for most of an hour.
    #[cfg(feature = "server-embedding")]
    pub async fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let key = crate::embeddings::policy::provider_key(&self.api_key_env, &url)?;
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": false,
        });
        let mut request = crate::embeddings::service::http()
            .post(&url)
            .timeout(std::time::Duration::from_secs(600))
            .json(&body);
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| Error::Internal(format!("the memory model at {url}: {e}")))?;
        let status = response.status();
        let value: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Internal(format!("the memory model at {url}: {e}")))?;
        if !status.is_success() {
            return Err(Error::Internal(format!(
                "the memory model at {url} answered {status}: {}",
                value
            )));
        }
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(strip_thinking(&content))
    }
}

#[cfg_attr(not(feature = "server-embedding"), allow(dead_code))]
fn strip_thinking(content: &str) -> String {
    match content.rfind("</think>") {
        Some(i) => content[i + "</think>".len()..].trim().to_string(),
        None => content.trim().to_string(),
    }
}

/// The date a relative time is resolved against: "10:37 am on 27 June, 2023" -> "27 June, 2023".
fn day_of(when: &str) -> &str {
    when.rsplit_once(" on ").map_or(when, |(_, day)| day).trim()
}

pub(crate) fn extract_prompt(text: &str, when: Option<&str>) -> String {
    let (held, rule) = match when {
        Some(when) => {
            let day = day_of(when);
            (
                format!(" It was held on {when}."),
                format!(
                    "- Never leave a relative time in a fact. Rewrite it against the date \
                     ({day}): \"yesterday\" -> \"the day before {day}\", \"last week\" -> \"the \
                     week before {day}\", \"last Friday\" -> \"the Friday before {day}\", \"last \
                     weekend\" -> \"the weekend before {day}\", \"last year\" -> the year \
                     before, \"ten years ago\" -> that year, \"next month\" -> that month. A fact \
                     with no time of its own happened on or just before {day}; say \"as of \
                     {day}\".\n"
                ),
            )
        }
        None => (String::new(), String::new()),
    };
    format!(
        "Below is a conversation.{held} List the facts it establishes about the people in it \
         that someone might later ask about: what they did, have, own, made, read, watched, \
         visited, like, feel, plan or decided; the people, pets, places, books, songs, events \
         and numbers involved, by name; and why, where it was said.\n\nRules:\n- One fact per \
         line, self-contained, starting with the person's name; no pronouns that need the \
         conversation to resolve.\n{rule}- Keep specific names, titles and counts exactly as \
         said. Do not invent anything.\n- Skip greetings and small talk.\n\nConversation:\n\
         {text}\n\nFacts, one per line:"
    )
}

/// The facts in a model's answer: one per line, list markers removed, fragments dropped.
pub(crate) fn parse_facts(reply: &str) -> Vec<String> {
    reply
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches(['-', '*', '•'])
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', ')'])
                .trim()
                .to_string()
        })
        .filter(|fact| fact.chars().count() > 12)
        .collect()
}

pub(crate) fn profile_prompt(name: &str, facts: &[String]) -> String {
    format!(
        "Below are facts collected about {name} over many conversations, each with its date. \
         Write {name}'s profile: everything someone would need to answer questions about \
         {name}, grouped under these headings, one line per item, keeping names, titles, counts \
         and dates exactly:\n\nIdentity and background; Family and relationships; Pets; Work, \
         business and career plans; Education; Health; Hobbies, sports and exercise; Creative \
         work; Books, films, TV, music and games (each one by title); Places visited and trips \
         (with dates); Events attended (with dates); Possessions and purchases; \
         Recommendations given and received (by title, and to or from whom); Goals, values and \
         beliefs; Opinions and personality as others describe it.\n\nMerge repeats; where a \
         fact changed, give the latest and note the change. Leave out a heading with nothing \
         under it. Do not invent anything.\n\nFacts:\n{}\n\nProfile of {name}:",
        facts.join("\n")
    )
}

/// Whether a fact is about `name`: the name as a word of it.
pub(crate) fn mentions(fact: &str, name: &str) -> bool {
    let name = name.to_lowercase();
    fact.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .any(|word| word == name)
}

/// The id of a person's profile under a set of tags: remade in place, never duplicated.
pub(crate) fn profile_id(name: &str, tags: &[String]) -> String {
    let mut tags = tags.to_vec();
    tags.sort();
    if tags.is_empty() {
        format!("profile:{name}")
    } else {
        format!("profile:{name}:{}", tags.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_times_are_resolved_against_the_day_not_the_clock() {
        assert_eq!(day_of("10:37 am on 27 June, 2023"), "27 June, 2023");
        assert_eq!(day_of("2023-06-27"), "2023-06-27");
        let prompt = extract_prompt("Caroline: hi", Some("10:37 am on 27 June, 2023"));
        assert!(
            prompt.contains("\"the Friday before 27 June, 2023\""),
            "{prompt}"
        );
        assert!(!extract_prompt("Caroline: hi", None).contains("Never leave"));
    }

    #[test]
    fn facts_are_lines_without_list_markers() {
        let reply = "1. Nate has three turtles as of 18 March 2022.\n- Nate adopted a dog named Max.\n\nok\n* Joanna wrote a screenplay about loss.";
        assert_eq!(
            parse_facts(reply),
            vec![
                "Nate has three turtles as of 18 March 2022.",
                "Nate adopted a dog named Max.",
                "Joanna wrote a screenplay about loss.",
            ]
        );
    }

    #[test]
    fn a_fact_is_about_a_person_when_it_names_them_as_a_word() {
        assert!(mentions("[27 June] Nate's dog is named Max", "Nate"));
        assert!(!mentions("Natalie has a cat", "Nate"));
    }

    #[test]
    fn a_profile_id_does_not_depend_on_tag_order() {
        assert_eq!(
            profile_id("Nate", &["b".into(), "a".into()]),
            profile_id("Nate", &["a".into(), "b".into()])
        );
        assert_eq!(profile_id("Nate", &[]), "profile:Nate");
    }

    #[test]
    fn a_thinking_block_is_not_the_answer() {
        assert_eq!(
            strip_thinking("<think>hmm</think>\nNate has a dog"),
            "Nate has a dog"
        );
    }
}
