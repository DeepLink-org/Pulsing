//! Tokenizer service for accurate token counting
//!
//! This module provides tokenizer functionality using HuggingFace tokenizers
//! for accurate input/output token counting.

use std::sync::Arc;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tracing::{info, warn};

/// Tokenizer service for token counting
#[derive(Clone)]
pub struct TokenizerService {
    tokenizer: Arc<Tokenizer>,
    model_name: String,
}

impl TokenizerService {
    /// Create a new tokenizer service from a HuggingFace model name
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
    /// * `hf_token` - Optional HuggingFace token for private models
    pub fn from_pretrained(model_name: &str, hf_token: Option<String>) -> anyhow::Result<Self> {
        info!("Loading tokenizer for model: {}", model_name);

        let params = FromPretrainedParameters {
            token: hf_token,
            ..Default::default()
        };

        let tokenizer = Tokenizer::from_pretrained(model_name, Some(params))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for {}: {}", model_name, e))?;

        info!("Tokenizer loaded successfully for {}", model_name);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            model_name: model_name.to_string(),
        })
    }

    /// Count tokens in a text string
    pub fn count_tokens(&self, text: &str) -> u64 {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_ids().len() as u64,
            Err(e) => {
                warn!("Failed to encode text: {}, using estimate", e);
                // Fallback to rough estimate
                (text.len() / 4) as u64
            }
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode: {}", e))
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// Fallback tokenizer that uses character-based estimation
/// when no real tokenizer is available
#[derive(Clone, Default)]
pub struct EstimateTokenizer;

impl EstimateTokenizer {
    /// Estimate token count based on word count
    /// Assumes roughly 1.3 tokens per word (common for English text)
    pub fn count_tokens(&self, text: &str) -> u64 {
        let spaces = text.chars().filter(|c| c.is_whitespace()).count();
        let words = spaces + 1;

        // Rough heuristic: ~1.3 tokens per word for English
        let estimate = (words as f64 * 1.3).ceil() as u64;

        // Ensure at least 1 token
        estimate.max(1)
    }
}

/// Token counter enum that can be either a real tokenizer or an estimator
#[derive(Clone)]
pub enum TokenCounter {
    /// Real HuggingFace tokenizer
    Real(TokenizerService),
    /// Estimation-based counter (fallback)
    Estimate(EstimateTokenizer),
}

impl TokenCounter {
    /// Create a real tokenizer
    pub fn real(model_name: &str, hf_token: Option<String>) -> anyhow::Result<Self> {
        Ok(Self::Real(TokenizerService::from_pretrained(
            model_name, hf_token,
        )?))
    }

    /// Create an estimate-based tokenizer
    pub fn estimate() -> Self {
        Self::Estimate(EstimateTokenizer)
    }

    /// Count tokens in text
    pub fn count_tokens(&self, text: &str) -> u64 {
        match self {
            Self::Real(t) => t.count_tokens(text),
            Self::Estimate(e) => e.count_tokens(text),
        }
    }

    /// Check if this is a real tokenizer
    pub fn is_real(&self) -> bool {
        matches!(self, Self::Real(_))
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::Estimate(EstimateTokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokenizer() {
        let estimator = EstimateTokenizer;

        // Simple English text (5 words -> ~7 tokens)
        let text = "Hello, how are you today?";
        let tokens = estimator.count_tokens(text);
        assert!(tokens >= 5);

        // Empty text (0 spaces -> 1 word -> ~2 tokens)
        let empty = "";
        let tokens = estimator.count_tokens(empty);
        assert!(tokens >= 1);

        // Long text (12 words -> ~16 tokens)
        let long_text = "This is a longer piece of text that should have more tokens.";
        let tokens = estimator.count_tokens(long_text);
        assert!(tokens > 10);
    }

    #[test]
    fn test_token_counter_default() {
        let counter = TokenCounter::default();
        assert!(!counter.is_real());

        let tokens = counter.count_tokens("Hello world");
        assert!(tokens > 0);
    }
}
