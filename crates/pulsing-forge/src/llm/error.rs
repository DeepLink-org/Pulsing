use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),

    #[error("missing API key for provider {0}")]
    MissingApiKey(String),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("api error ({status}): {body}")]
    Api { status: u16, body: String },

    #[error("stream error: {0}")]
    Stream(String),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl LlmError {
    pub fn is_authentication_error(&self) -> bool {
        matches!(
            self,
            Self::Api {
                status: 401 | 403,
                ..
            }
        )
    }

    pub fn is_retryable_error(&self) -> bool {
        match self {
            Self::Http(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            Self::Api { status, .. } => matches!(status, 408 | 429 | 500..=599),
            Self::Stream(_) => true,
            _ => false,
        }
    }

    pub fn is_api_error(&self) -> bool {
        matches!(self, Self::Api { .. })
    }
}
