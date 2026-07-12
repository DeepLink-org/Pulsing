//! OAuth token storage and login flow (Codex `rmcp-client/oauth.rs` subset).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::codex_home::credentials_path;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StoredOAuthTokens {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CredentialsFile {
    #[serde(default)]
    mcp_oauth_tokens: HashMap<String, StoredOAuthTokens>,
}

#[derive(Clone, Default)]
pub struct OAuthCredentialsStore {
    path: PathBuf,
    inner: Arc<Mutex<CredentialsFile>>,
}

impl OAuthCredentialsStore {
    pub fn load_default() -> Self {
        Self::load(credentials_path())
    }

    pub fn load(path: PathBuf) -> Self {
        let inner = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            path,
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn get(&self, server_name: &str) -> Option<StoredOAuthTokens> {
        self.inner
            .lock()
            .ok()
            .and_then(|f| f.mcp_oauth_tokens.get(server_name).cloned())
    }

    pub fn save(&self, server_name: &str, tokens: StoredOAuthTokens) -> std::io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| std::io::Error::other("credentials lock poisoned"))?;
        guard
            .mcp_oauth_tokens
            .insert(server_name.to_string(), tokens);
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&*guard)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(&self.path, text)
    }

    pub fn delete(&self, server_name: &str) -> std::io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| std::io::Error::other("credentials lock poisoned"))?;
        guard.mcp_oauth_tokens.remove(server_name);
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&*guard)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(&self.path, text)
    }
}

/// Placeholder for Codex `perform_oauth_login_return_url` — returns URL for host UI.
pub struct OAuthLoginHandle {
    pub authorization_url: String,
    pub server_name: String,
}

pub fn perform_oauth_login(server_name: &str, authorization_url: String) -> OAuthLoginHandle {
    OAuthLoginHandle {
        authorization_url,
        server_name: server_name.to_string(),
    }
}
