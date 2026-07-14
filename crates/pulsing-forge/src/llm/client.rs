use reqwest::Client;

use super::anthropic::AnthropicStream;
use super::demo::DemoStream;
use super::error::LlmError;
use super::openai::OpenAiStream;
use super::types::{LlmMessage, Provider, StreamRequest};

#[derive(Debug, Clone)]
pub struct LlmClientConfig {
    pub provider: Provider,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Clone)]
pub struct LlmClient {
    config: LlmClientConfig,
    http: Client,
}

pub enum LlmStream {
    Demo(DemoStream),
    OpenAi(OpenAiStream),
    Anthropic(AnthropicStream),
}

impl LlmStream {
    pub fn text_chunks(&self) -> Vec<String> {
        match self {
            Self::Demo(s) => s.text_chunks(),
            Self::OpenAi(s) => s.text_chunks().to_vec(),
            Self::Anthropic(s) => s.text_chunks().to_vec(),
        }
    }

    pub fn final_message(&self) -> LlmMessage {
        match self {
            Self::Demo(s) => s.final_message(),
            Self::OpenAi(s) => s.final_message(),
            Self::Anthropic(s) => s.final_message(),
        }
    }
}

impl LlmClient {
    pub fn new(
        provider: &str,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, LlmError> {
        let provider = Provider::parse(provider)
            .ok_or_else(|| LlmError::UnsupportedProvider(provider.to_string()))?;
        let http = Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        Ok(Self {
            config: LlmClientConfig {
                provider,
                api_key,
                base_url,
            },
            http,
        })
    }

    pub fn provider(&self) -> Provider {
        self.config.provider
    }

    pub async fn stream_messages(&self, req: StreamRequest) -> Result<LlmStream, LlmError> {
        match self.config.provider {
            Provider::Demo => Ok(LlmStream::Demo(DemoStream::new(&req.messages, &req.tools))),
            Provider::Openai => {
                let api_key = self.resolve_api_key("OPENAI_API_KEY")?;
                let base = self
                    .config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
                let stream = OpenAiStream::start(&self.http, &api_key, &base, &req).await?;
                Ok(LlmStream::OpenAi(stream))
            }
            Provider::Anthropic => {
                let api_key = self.resolve_api_key("ANTHROPIC_API_KEY")?;
                let base = self
                    .config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
                let stream = AnthropicStream::start(&self.http, &api_key, &base, &req).await?;
                Ok(LlmStream::Anthropic(stream))
            }
        }
    }

    fn resolve_api_key(&self, env_name: &str) -> Result<String, LlmError> {
        if let Some(key) = self.config.api_key.as_deref().filter(|k| !k.is_empty()) {
            return Ok(key.to_string());
        }
        if let Ok(key) = std::env::var(env_name)
            && !key.is_empty()
        {
            return Ok(key);
        }
        Err(LlmError::MissingApiKey(
            self.config.provider.as_str().into(),
        ))
    }

    pub fn classify_error(err: &LlmError) -> (bool, bool, bool) {
        (
            err.is_authentication_error(),
            err.is_retryable_error(),
            err.is_api_error(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn demo_stream_glob_reply() {
        let client = LlmClient::new("demo", None, None).expect("demo client");
        let req = StreamRequest {
            model: "demo".into(),
            max_tokens: 1024,
            messages: vec![json!({ "role": "user", "content": "list files with Glob" })],
            system: None,
            tools: vec![json!({ "name": "Glob", "input_schema": {} })],
        };
        let stream = client.stream_messages(req).await.expect("stream");
        let msg = stream.final_message();
        assert_eq!(
            msg.content[0].get("type").and_then(|v| v.as_str()),
            Some("tool_use")
        );
    }
}
