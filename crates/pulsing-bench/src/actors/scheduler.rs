//! Scheduler Actor - controls request timing and rate

use super::messages::*;
use crate::tokenizer::TokenCounter;
use async_trait::async_trait;
use pulsing_actor::error::{PulsingError, RuntimeError};
use pulsing_actor::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{info, warn};

/// Request generator trait
pub trait RequestGenerator: Send + Sync {
    fn generate(&self) -> RequestTemplate;
}

/// Simple round-robin request generator (estimation-based)
pub struct SimpleRequestGenerator {
    prompts: Vec<String>,
    index: AtomicU64,
}

impl SimpleRequestGenerator {
    pub fn new(prompts: Vec<String>) -> Self {
        Self {
            prompts,
            index: AtomicU64::new(0),
        }
    }

    pub fn default_prompts() -> Self {
        Self::new(vec![
            "What is the meaning of life?".to_string(),
            "Explain quantum computing in simple terms.".to_string(),
            "Write a short poem about the ocean.".to_string(),
            "What are the benefits of exercise?".to_string(),
            "How does photosynthesis work?".to_string(),
        ])
    }
}

impl RequestGenerator for SimpleRequestGenerator {
    fn generate(&self) -> RequestTemplate {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) as usize;
        let prompt = &self.prompts[idx % self.prompts.len()];
        RequestTemplate {
            id: format!("req-{}", idx),
            prompt: prompt.clone(),
            num_prompt_tokens: (prompt.len() / 4) as u64, // rough estimate
            num_decode_tokens: Some(100),
        }
    }
}

/// Tokenized request generator - uses real tokenizer for accurate token counting
pub struct TokenizedRequestGenerator {
    prompts: Vec<String>,
    index: AtomicU64,
    token_counter: TokenCounter,
}

impl TokenizedRequestGenerator {
    pub fn new(token_counter: TokenCounter) -> Self {
        Self {
            prompts: vec![
                "What is the meaning of life?".to_string(),
                "Explain quantum computing in simple terms.".to_string(),
                "Write a short poem about the ocean.".to_string(),
                "What are the benefits of exercise?".to_string(),
                "How does photosynthesis work?".to_string(),
                "Describe the process of machine learning.".to_string(),
                "What causes the northern lights?".to_string(),
                "Explain how a computer works.".to_string(),
            ],
            index: AtomicU64::new(0),
            token_counter,
        }
    }

    pub fn with_prompts(mut self, prompts: Vec<String>) -> Self {
        self.prompts = prompts;
        self
    }
}

impl RequestGenerator for TokenizedRequestGenerator {
    fn generate(&self) -> RequestTemplate {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) as usize;
        let prompt = &self.prompts[idx % self.prompts.len()];

        // Use real tokenizer for accurate token counting
        let num_prompt_tokens = self.token_counter.count_tokens(prompt);

        RequestTemplate {
            id: format!("req-{}", idx),
            prompt: prompt.clone(),
            num_prompt_tokens,
            num_decode_tokens: Some(100),
        }
    }
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub scheduler_type: SchedulerType,
    pub max_vus: u64,
    pub duration: Duration,
    pub rate: Option<f64>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::ConstantVUs,
            max_vus: 10,
            duration: Duration::from_secs(60),
            rate: None,
        }
    }
}

/// Scheduler Actor - manages request scheduling
pub struct SchedulerActor {
    /// Current configuration
    config: SchedulerConfig,
    /// Request generator
    request_gen: Arc<dyn RequestGenerator>,
    /// Worker actor references
    workers: Vec<ActorRef>,
    /// Coordinator reference for progress updates
    coordinator_ref: Option<ActorRef>,
    /// MetricsAggregator reference
    metrics_ref: Option<ActorRef>,
    /// Target URL
    target_url: String,
    /// API key
    api_key: String,
    /// Model name
    model_name: String,
    /// Is scheduling active
    is_active: Arc<AtomicBool>,
    /// Current phase ID
    phase_id: String,
    /// Sent requests counter
    sent_requests: Arc<AtomicU64>,
    /// Active VUs counter
    active_vus: Arc<AtomicU64>,
    /// Start time
    start_time: Option<Instant>,
}

impl SchedulerActor {
    pub fn new() -> Self {
        Self {
            config: SchedulerConfig::default(),
            request_gen: Arc::new(SimpleRequestGenerator::default_prompts()),
            workers: Vec::new(),
            coordinator_ref: None,
            metrics_ref: None,
            target_url: "http://localhost:8000".to_string(),
            api_key: String::new(),
            model_name: "gpt2".to_string(),
            is_active: Arc::new(AtomicBool::new(false)),
            phase_id: String::new(),
            sent_requests: Arc::new(AtomicU64::new(0)),
            active_vus: Arc::new(AtomicU64::new(0)),
            start_time: None,
        }
    }

    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_request_generator(mut self, gen: Arc<dyn RequestGenerator>) -> Self {
        self.request_gen = gen;
        self
    }

    pub fn with_workers(mut self, workers: Vec<ActorRef>) -> Self {
        self.workers = workers;
        self
    }

    pub fn with_coordinator(mut self, coordinator: ActorRef) -> Self {
        self.coordinator_ref = Some(coordinator);
        self
    }

    pub fn with_metrics(mut self, metrics: ActorRef) -> Self {
        self.metrics_ref = Some(metrics);
        self
    }

    pub fn with_target(mut self, url: String, api_key: String, model_name: String) -> Self {
        self.target_url = url;
        self.api_key = api_key;
        self.model_name = model_name;
        self
    }

    fn configure(&mut self, config: ConfigureScheduler) {
        self.config = SchedulerConfig {
            scheduler_type: config.scheduler_type,
            max_vus: config.max_vus,
            duration: Duration::from_secs(config.duration_secs),
            rate: config.rate,
        };
        info!("Scheduler configured: {:?}", self.config);
    }

    async fn start_scheduling(&mut self, phase_id: String) -> anyhow::Result<()> {
        if self.workers.is_empty() {
            return Err(anyhow::anyhow!("No workers configured"));
        }

        self.phase_id = phase_id.clone();
        self.is_active.store(true, Ordering::SeqCst);
        self.sent_requests.store(0, Ordering::SeqCst);
        self.active_vus.store(0, Ordering::SeqCst);
        self.start_time = Some(Instant::now());

        info!(
            "Starting scheduling for phase {} with {:?}",
            phase_id, self.config.scheduler_type
        );

        match self.config.scheduler_type {
            SchedulerType::ConstantVUs => {
                self.run_constant_vus().await?;
            }
            SchedulerType::ConstantArrivalRate => {
                self.run_constant_rate().await?;
            }
        }

        Ok(())
    }

    /// Run constant VUs scheduling - maintain N concurrent requests
    async fn run_constant_vus(&mut self) -> anyhow::Result<()> {
        let max_vus = self.config.max_vus;
        let duration = self.config.duration;
        let is_active = self.is_active.clone();
        let sent_requests = self.sent_requests.clone();
        let active_vus = self.active_vus.clone();

        let start = Instant::now();
        let mut handles = Vec::new();

        // Spawn initial VUs
        for i in 0..max_vus {
            if !is_active.load(Ordering::SeqCst) {
                break;
            }

            let worker_idx = i as usize % self.workers.len();
            let worker = self.workers[worker_idx].clone();
            let template = self.request_gen.generate();
            let request_id = template.id.clone();

            let request = SendRequest {
                request_id,
                template,
                target_url: self.target_url.clone(),
                api_key: self.api_key.clone(),
                model_name: self.model_name.clone(),
            };

            sent_requests.fetch_add(1, Ordering::SeqCst);
            active_vus.fetch_add(1, Ordering::SeqCst);

            let is_active_clone = is_active.clone();
            let active_vus_clone = active_vus.clone();
            let sent_requests_clone = sent_requests.clone();
            let workers = self.workers.clone();
            let request_gen = self.request_gen.clone();
            let target_url = self.target_url.clone();
            let api_key = self.api_key.clone();
            let model_name = self.model_name.clone();

            let handle = tokio::spawn(async move {
                // Send initial request
                let _: RequestCompleted = worker.ask(request).await.unwrap_or_default();
                active_vus_clone.fetch_sub(1, Ordering::SeqCst);

                // Keep spawning requests while duration not reached
                while is_active_clone.load(Ordering::SeqCst) && start.elapsed() < duration {
                    let worker_idx =
                        sent_requests_clone.load(Ordering::SeqCst) as usize % workers.len();
                    let worker = &workers[worker_idx];
                    let template = request_gen.generate();

                    let request = SendRequest {
                        request_id: template.id.clone(),
                        template,
                        target_url: target_url.clone(),
                        api_key: api_key.clone(),
                        model_name: model_name.clone(),
                    };

                    sent_requests_clone.fetch_add(1, Ordering::SeqCst);
                    active_vus_clone.fetch_add(1, Ordering::SeqCst);

                    let _: RequestCompleted = worker.ask(request).await.unwrap_or_default();
                    active_vus_clone.fetch_sub(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        // Wait for duration or until stopped
        tokio::select! {
            _ = tokio::time::sleep(duration) => {
                info!("Duration reached, stopping scheduling");
            }
            _ = async {
                while is_active.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            } => {
                info!("Scheduling stopped externally");
            }
        }

        is_active.store(false, Ordering::SeqCst);

        // Wait for all VUs to complete
        for handle in handles {
            let _ = handle.await;
        }

        info!(
            "Constant VUs scheduling completed. Sent: {} requests",
            sent_requests.load(Ordering::SeqCst)
        );

        Ok(())
    }

    /// Run constant arrival rate scheduling - send requests at fixed rate
    async fn run_constant_rate(&mut self) -> anyhow::Result<()> {
        let rate = self
            .config
            .rate
            .ok_or_else(|| anyhow::anyhow!("Rate not configured"))?;
        let max_vus = self.config.max_vus;
        let duration = self.config.duration;
        let is_active = self.is_active.clone();
        let sent_requests = self.sent_requests.clone();
        let active_vus = self.active_vus.clone();

        let start = Instant::now();

        // Calculate tick interval to achieve target rate
        let tick_ms = 10u64;
        let requests_per_tick = rate * (tick_ms as f64) / 1000.0;
        let mut spawn_queue = 0.0;

        let mut ticker = interval(Duration::from_millis(tick_ms));

        while start.elapsed() < duration && is_active.load(Ordering::SeqCst) {
            ticker.tick().await;

            spawn_queue += requests_per_tick;

            // Spawn requests when we have accumulated enough
            while spawn_queue >= 1.0 {
                // Check VU limit
                if active_vus.load(Ordering::SeqCst) >= max_vus {
                    warn!("Max VUs reached, skipping request");
                    spawn_queue -= 1.0;
                    continue;
                }

                let worker_idx = sent_requests.load(Ordering::SeqCst) as usize % self.workers.len();
                let worker = self.workers[worker_idx].clone();
                let template = self.request_gen.generate();

                let request = SendRequest {
                    request_id: template.id.clone(),
                    template,
                    target_url: self.target_url.clone(),
                    api_key: self.api_key.clone(),
                    model_name: self.model_name.clone(),
                };

                sent_requests.fetch_add(1, Ordering::SeqCst);
                active_vus.fetch_add(1, Ordering::SeqCst);

                let active_vus_clone = active_vus.clone();

                // Fire and forget - don't wait for response
                tokio::spawn(async move {
                    let _: RequestCompleted = worker.ask(request).await.unwrap_or_default();
                    active_vus_clone.fetch_sub(1, Ordering::SeqCst);
                });

                spawn_queue -= 1.0;
            }
        }

        is_active.store(false, Ordering::SeqCst);

        // Wait for remaining requests to complete
        while active_vus.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!(
            "Constant rate scheduling completed. Sent: {} requests at {:.2} req/s",
            sent_requests.load(Ordering::SeqCst),
            rate
        );

        Ok(())
    }

    fn get_progress(&self) -> SchedulerProgress {
        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let progress = if self.config.duration.as_secs_f64() > 0.0 {
            (elapsed / self.config.duration.as_secs_f64() * 100.0).min(100.0)
        } else {
            0.0
        };

        SchedulerProgress {
            phase_id: self.phase_id.clone(),
            progress_pct: progress,
            sent_requests: self.sent_requests.load(Ordering::SeqCst),
            active_vus: self.active_vus.load(Ordering::SeqCst),
            elapsed_secs: elapsed,
        }
    }
}

impl Default for SchedulerActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for SchedulerActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        info!("Scheduler started with actor_id {:?}", ctx.id());
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        self.is_active.store(false, Ordering::SeqCst);
        info!(
            "Scheduler stopped. Total sent: {}",
            self.sent_requests.load(Ordering::SeqCst)
        );
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("ConfigureScheduler") {
            let config: ConfigureScheduler = msg.unpack()?;
            self.configure(config);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("StartScheduling") {
            let start: StartScheduling = msg.unpack()?;
            match self.start_scheduling(start.phase_id).await {
                Ok(()) => return Message::pack(&AckMessage::ok()),
                Err(e) => return Message::pack(&AckMessage::error(e.to_string())),
            }
        }

        if msg_type.ends_with("PauseScheduling") {
            self.is_active.store(false, Ordering::SeqCst);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("ResumeScheduling") {
            // Note: Full resume would need to restart the scheduling loop
            self.is_active.store(true, Ordering::SeqCst);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("SchedulerProgress") || msg_type.ends_with("GetProgress") {
            return Message::pack(&self.get_progress());
        }

        Err(PulsingError::from(RuntimeError::Other(format!(
            "Unknown message type: {}",
            msg_type
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_request_generator() {
        let gen = SimpleRequestGenerator::default_prompts();
        let req1 = gen.generate();
        let req2 = gen.generate();
        assert_ne!(req1.id, req2.id);
    }

    #[test]
    fn test_scheduler_config() {
        let config = SchedulerConfig {
            scheduler_type: SchedulerType::ConstantArrivalRate,
            max_vus: 100,
            duration: Duration::from_secs(60),
            rate: Some(10.0),
        };
        assert_eq!(config.rate, Some(10.0));
    }

    #[test]
    fn test_simple_request_generator_custom_prompts() {
        let prompts = vec!["Hello".to_string(), "World".to_string()];
        let gen = SimpleRequestGenerator::new(prompts);

        let req1 = gen.generate();
        let _req2 = gen.generate(); // consume second request
        let req3 = gen.generate();

        // Third request should wrap around to first prompt
        assert!(req1.prompt == "Hello" || req1.prompt == "World");
        assert!(req3.prompt == "Hello" || req3.prompt == "World");
    }

    #[test]
    fn test_simple_request_generator_id_increment() {
        let gen = SimpleRequestGenerator::default_prompts();

        let req1 = gen.generate();
        let req2 = gen.generate();
        let req3 = gen.generate();

        assert_eq!(req1.id, "req-0");
        assert_eq!(req2.id, "req-1");
        assert_eq!(req3.id, "req-2");
    }

    #[test]
    fn test_simple_request_generator_token_estimate() {
        let prompts = vec!["Hello world test".to_string()];
        let gen = SimpleRequestGenerator::new(prompts);
        let req = gen.generate();

        // Token estimate is len / 4
        // "Hello world test" = 16 chars / 4 = 4 tokens
        assert_eq!(req.num_prompt_tokens, 4);
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.scheduler_type, SchedulerType::ConstantVUs);
        assert_eq!(config.max_vus, 10);
        assert_eq!(config.duration, Duration::from_secs(60));
        assert!(config.rate.is_none());
    }

    #[test]
    fn test_scheduler_actor_new() {
        let scheduler = SchedulerActor::new();
        assert!(scheduler.workers.is_empty());
        assert!(scheduler.coordinator_ref.is_none());
        assert!(scheduler.metrics_ref.is_none());
        assert_eq!(scheduler.target_url, "http://localhost:8000");
        assert!(scheduler.api_key.is_empty());
        assert_eq!(scheduler.model_name, "gpt2");
    }

    #[test]
    fn test_scheduler_actor_default() {
        let scheduler = SchedulerActor::default();
        assert!(scheduler.workers.is_empty());
    }

    #[test]
    fn test_scheduler_actor_with_config() {
        let config = SchedulerConfig {
            scheduler_type: SchedulerType::ConstantArrivalRate,
            max_vus: 50,
            duration: Duration::from_secs(120),
            rate: Some(20.0),
        };

        let scheduler = SchedulerActor::new().with_config(config.clone());
        assert_eq!(
            scheduler.config.scheduler_type,
            SchedulerType::ConstantArrivalRate
        );
        assert_eq!(scheduler.config.max_vus, 50);
        assert_eq!(scheduler.config.rate, Some(20.0));
    }

    #[test]
    fn test_scheduler_actor_with_target() {
        let scheduler = SchedulerActor::new().with_target(
            "http://example.com".to_string(),
            "api_key".to_string(),
            "llama".to_string(),
        );

        assert_eq!(scheduler.target_url, "http://example.com");
        assert_eq!(scheduler.api_key, "api_key");
        assert_eq!(scheduler.model_name, "llama");
    }

    #[test]
    fn test_scheduler_progress_initial() {
        let scheduler = SchedulerActor::new();
        let progress = scheduler.get_progress();

        assert!(progress.phase_id.is_empty());
        assert_eq!(progress.progress_pct, 0.0);
        assert_eq!(progress.sent_requests, 0);
        assert_eq!(progress.active_vus, 0);
        assert_eq!(progress.elapsed_secs, 0.0);
    }

    #[test]
    fn test_tokenized_request_generator() {
        let token_counter = TokenCounter::estimate();
        let gen = TokenizedRequestGenerator::new(token_counter);

        let req = gen.generate();
        assert!(!req.prompt.is_empty());
        assert!(req.num_prompt_tokens > 0);
        assert_eq!(req.id, "req-0");
    }

    #[test]
    fn test_tokenized_request_generator_with_prompts() {
        let token_counter = TokenCounter::estimate();
        let prompts = vec![
            "Custom prompt one".to_string(),
            "Custom prompt two".to_string(),
        ];
        let gen = TokenizedRequestGenerator::new(token_counter).with_prompts(prompts);

        let req1 = gen.generate();
        let req2 = gen.generate();

        assert!(req1.prompt == "Custom prompt one" || req1.prompt == "Custom prompt two");
        assert!(req2.prompt == "Custom prompt one" || req2.prompt == "Custom prompt two");
    }

    #[test]
    fn test_tokenized_request_generator_id_increment() {
        let token_counter = TokenCounter::estimate();
        let gen = TokenizedRequestGenerator::new(token_counter);

        let req1 = gen.generate();
        let req2 = gen.generate();
        let req3 = gen.generate();

        assert_eq!(req1.id, "req-0");
        assert_eq!(req2.id, "req-1");
        assert_eq!(req3.id, "req-2");
    }

    #[test]
    fn test_scheduler_configure() {
        let mut scheduler = SchedulerActor::new();

        let config_msg = ConfigureScheduler {
            scheduler_type: SchedulerType::ConstantArrivalRate,
            max_vus: 100,
            duration_secs: 180,
            rate: Some(50.0),
        };

        scheduler.configure(config_msg);

        assert_eq!(
            scheduler.config.scheduler_type,
            SchedulerType::ConstantArrivalRate
        );
        assert_eq!(scheduler.config.max_vus, 100);
        assert_eq!(scheduler.config.duration, Duration::from_secs(180));
        assert_eq!(scheduler.config.rate, Some(50.0));
    }

    #[test]
    fn test_request_template_decode_tokens() {
        let gen = SimpleRequestGenerator::default_prompts();
        let req = gen.generate();

        // Default decode tokens should be 100
        assert_eq!(req.num_decode_tokens, Some(100));
    }
}
