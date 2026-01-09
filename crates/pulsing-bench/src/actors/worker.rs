//! Worker Actor - responsible for sending HTTP requests

use super::messages::*;
use async_trait::async_trait;
use futures_util::StreamExt;
use pulsing_actor::prelude::*;
use reqwest::Client;
use reqwest_eventsource::{Event as SseEvent, EventSource};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, trace};

/// Worker Actor - sends HTTP requests and reports results
pub struct WorkerActor {
    /// Worker ID
    worker_id: String,
    /// HTTP client
    client: Client,
    /// MetricsAggregator actor reference for reporting results
    metrics_ref: Option<ActorRef>,
    /// Active request count
    active_requests: Arc<AtomicU32>,
    /// Completed request count
    completed_requests: Arc<AtomicU64>,
    /// Failed request count
    failed_requests: Arc<AtomicU64>,
    /// Request timeout in seconds
    timeout_secs: u64,
}

impl WorkerActor {
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            client: Client::builder()
                .pool_max_idle_per_host(100)
                .build()
                .expect("Failed to create HTTP client"),
            metrics_ref: None,
            active_requests: Arc::new(AtomicU32::new(0)),
            completed_requests: Arc::new(AtomicU64::new(0)),
            failed_requests: Arc::new(AtomicU64::new(0)),
            timeout_secs: 300,
        }
    }

    pub fn with_metrics(mut self, metrics: ActorRef) -> Self {
        self.metrics_ref = Some(metrics);
        self
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Send a streaming request to the LLM endpoint
    async fn send_request(&self, req: SendRequest) -> RequestCompleted {
        let start_time = Instant::now();
        self.active_requests.fetch_add(1, Ordering::SeqCst);

        let result = self.execute_streaming_request(&req, start_time).await;

        self.active_requests.fetch_sub(1, Ordering::SeqCst);

        match result {
            Ok(completed) => {
                self.completed_requests.fetch_add(1, Ordering::SeqCst);
                completed
            }
            Err(e) => {
                self.failed_requests.fetch_add(1, Ordering::SeqCst);
                RequestCompleted {
                    request_id: req.request_id,
                    success: false,
                    error: Some(e.to_string()),
                    ttft_ms: None,
                    latency_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                    generated_tokens: 0,
                    prompt_tokens: req.template.num_prompt_tokens,
                    tpot_ms: None,
                    token_times_ms: vec![],
                }
            }
        }
    }

    async fn execute_streaming_request(
        &self,
        req: &SendRequest,
        start_time: Instant,
    ) -> anyhow::Result<RequestCompleted> {
        let url = format!("{}/v1/chat/completions", req.target_url);

        let body = serde_json::json!({
            "model": req.model_name,
            "messages": [
                {"role": "user", "content": req.template.prompt}
            ],
            "stream": true,
            "max_tokens": req.template.num_decode_tokens.unwrap_or(100),
        });

        let request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        let request = if !req.api_key.is_empty() {
            request.header("Authorization", format!("Bearer {}", req.api_key))
        } else {
            request
        };

        let request = request.json(&body);

        let mut es = EventSource::new(request)?;
        let mut first_token_time: Option<Instant> = None;
        let mut token_times: Vec<f64> = Vec::new();
        let mut generated_tokens: u64 = 0;
        let mut response_text = String::new();

        while let Some(event) = es.next().await {
            match event {
                Ok(SseEvent::Open) => {
                    trace!("SSE connection opened for request {}", req.request_id);
                }
                Ok(SseEvent::Message(message)) => {
                    let data = message.data.trim();

                    if data == "[DONE]" {
                        break;
                    }

                    // Parse the SSE data
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                            if let Some(choice) = choices.first() {
                                // Check for content delta
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) =
                                        delta.get("content").and_then(|c| c.as_str())
                                    {
                                        let now = Instant::now();

                                        if first_token_time.is_none() {
                                            first_token_time = Some(now);
                                        }

                                        token_times.push(
                                            now.duration_since(start_time).as_secs_f64() * 1000.0,
                                        );
                                        generated_tokens += 1;
                                        response_text.push_str(content);
                                    }
                                }

                                // Check for finish reason
                                if let Some(finish_reason) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    if finish_reason == "stop" || finish_reason == "length" {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("SSE error: {}", e));
                }
            }
        }

        let end_time = Instant::now();
        let latency_ms = end_time.duration_since(start_time).as_secs_f64() * 1000.0;

        let ttft_ms = first_token_time.map(|t| t.duration_since(start_time).as_secs_f64() * 1000.0);

        // Calculate TPOT (time per output token) - only for tokens after the first
        let tpot_ms = if token_times.len() > 1 {
            let decode_time = latency_ms - ttft_ms.unwrap_or(0.0);
            Some(decode_time / (generated_tokens as f64 - 1.0).max(1.0))
        } else {
            None
        };

        Ok(RequestCompleted {
            request_id: req.request_id.clone(),
            success: true,
            error: None,
            ttft_ms,
            latency_ms,
            generated_tokens,
            prompt_tokens: req.template.num_prompt_tokens,
            tpot_ms,
            token_times_ms: token_times,
        })
    }
}

#[async_trait]
impl Actor for WorkerActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        info!(
            "Worker {} started with actor_id {:?}",
            self.worker_id,
            ctx.id()
        );
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        info!(
            "Worker {} stopped. Completed: {}, Failed: {}",
            self.worker_id,
            self.completed_requests.load(Ordering::SeqCst),
            self.failed_requests.load(Ordering::SeqCst)
        );
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("SendRequest") {
            let request: SendRequest = msg.unpack()?;
            debug!(
                "Worker {} received request {}",
                self.worker_id, request.request_id
            );

            let result = self.send_request(request).await;

            // Report to metrics aggregator if configured
            if let Some(ref metrics) = self.metrics_ref {
                let _ = metrics.tell(result.clone()).await;
            }

            return Message::pack(&result);
        }

        if msg_type.ends_with("WorkerStatus") {
            let status = WorkerStatus {
                worker_id: self.worker_id.clone(),
                active_requests: self.active_requests.load(Ordering::SeqCst),
                completed_requests: self.completed_requests.load(Ordering::SeqCst),
                failed_requests: self.failed_requests.load(Ordering::SeqCst),
            };
            return Message::pack(&status);
        }

        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: This test requires network/system access for HTTP client creation
    // It may fail in sandboxed environments
    #[test]
    #[ignore]
    fn test_worker_creation() {
        let worker = WorkerActor::new("test-worker");
        assert_eq!(worker.worker_id, "test-worker");
        assert_eq!(worker.active_requests.load(Ordering::SeqCst), 0);
    }
}
