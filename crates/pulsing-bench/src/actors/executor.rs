//! ExecutorActor - Sends requests and collects responses
//!
//! Responsible for:
//! - Managing virtual users (VUs)
//! - Sending requests at configured rate
//! - Forwarding results to CollectorActor

use super::messages::*;
use crate::requests::{
    TextGenerationBackend, TextRequestGenerator,
};
use async_trait::async_trait;
use pulsing_actor::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::info;

/// ExecutorActor manages request execution
pub struct ExecutorActor {
    /// Backend for sending requests
    backend: Arc<dyn TextGenerationBackend + Send + Sync>,
    /// Request generator
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    /// Current configuration
    config: Option<StartExecutor>,
    /// Running state
    is_running: Arc<AtomicBool>,
    /// Request counter
    request_counter: Arc<AtomicU64>,
    /// Active VUs
    active_vus: Arc<AtomicU64>,
    /// Successful requests
    successful_requests: Arc<AtomicU64>,
    /// Failed requests
    failed_requests: Arc<AtomicU64>,
    /// Results storage (request_id -> result)
    pub results: Arc<RwLock<Vec<RequestResult>>>,
    /// Start time
    start_time: Option<Instant>,
    /// Collector actor reference
    collector_ref: Option<ActorRef>,
}

impl ExecutorActor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    ) -> Self {
        Self {
            backend: Arc::from(backend),
            requests,
            config: None,
            is_running: Arc::new(AtomicBool::new(false)),
            request_counter: Arc::new(AtomicU64::new(0)),
            active_vus: Arc::new(AtomicU64::new(0)),
            successful_requests: Arc::new(AtomicU64::new(0)),
            failed_requests: Arc::new(AtomicU64::new(0)),
            results: Arc::new(RwLock::new(Vec::new())),
            start_time: None,
            collector_ref: None,
        }
    }

    /// Start constant VUs execution
    pub async fn run_constant_vus(&mut self, config: StartExecutor) -> anyhow::Result<ExecutorFinished> {
        info!("Starting constant VUs executor: {} VUs for {} seconds", config.max_vus, config.duration_secs);
        
        self.is_running.store(true, Ordering::SeqCst);
        self.start_time = Some(Instant::now());
        
        let duration = Duration::from_secs(config.duration_secs);
        let max_vus = config.max_vus;
        
        // Channel for VU completion signals
        let (end_tx, mut end_rx) = mpsc::channel::<()>(max_vus as usize);
        
        // Start initial VUs
        for _ in 0..max_vus {
            self.spawn_vu(end_tx.clone()).await;
        }
        
        // Replenish VUs as they complete
        let start = Instant::now();
        while start.elapsed() < duration && self.is_running.load(Ordering::SeqCst) {
            tokio::select! {
                _ = end_rx.recv() => {
                    self.active_vus.fetch_sub(1, Ordering::SeqCst);
                    if start.elapsed() < duration {
                        self.spawn_vu(end_tx.clone()).await;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }
        
        // Wait for remaining VUs
        self.is_running.store(false, Ordering::SeqCst);
        while self.active_vus.load(Ordering::SeqCst) > 0 {
            if end_rx.recv().await.is_some() {
                self.active_vus.fetch_sub(1, Ordering::SeqCst);
            }
        }
        
        Ok(ExecutorFinished {
            id: config.id,
            total_requests: self.successful_requests.load(Ordering::SeqCst) 
                + self.failed_requests.load(Ordering::SeqCst),
            successful_requests: self.successful_requests.load(Ordering::SeqCst),
            failed_requests: self.failed_requests.load(Ordering::SeqCst),
        })
    }

    /// Start constant arrival rate execution
    pub async fn run_constant_arrival_rate(&mut self, config: StartExecutor) -> anyhow::Result<ExecutorFinished> {
        let rate = config.rate.ok_or_else(|| anyhow::anyhow!("Rate required for constant arrival rate"))?;
        info!("Starting constant arrival rate executor: {} req/s for {} seconds", rate, config.duration_secs);
        
        self.is_running.store(true, Ordering::SeqCst);
        self.start_time = Some(Instant::now());
        
        let duration = Duration::from_secs(config.duration_secs);
        let max_vus = config.max_vus;
        
        // Channel for VU completion signals
        let (end_tx, mut end_rx) = mpsc::channel::<()>(max_vus as usize * 10);
        
        // Spawn rate controller
        let tick_ms = 10u64;
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
        let mut spawn_queue = 0.0f64;
        
        let start = Instant::now();
        while start.elapsed() < duration && self.is_running.load(Ordering::SeqCst) {
            tokio::select! {
                _ = interval.tick() => {
                    spawn_queue += rate * (tick_ms as f64) / 1000.0;
                    
                    while spawn_queue >= 1.0 && self.active_vus.load(Ordering::SeqCst) < max_vus {
                        self.spawn_vu(end_tx.clone()).await;
                        spawn_queue -= 1.0;
                    }
                }
                _ = end_rx.recv() => {
                    self.active_vus.fetch_sub(1, Ordering::SeqCst);
                }
            }
        }
        
        // Wait for remaining VUs
        self.is_running.store(false, Ordering::SeqCst);
        while self.active_vus.load(Ordering::SeqCst) > 0 {
            if end_rx.recv().await.is_some() {
                self.active_vus.fetch_sub(1, Ordering::SeqCst);
            }
        }
        
        Ok(ExecutorFinished {
            id: config.id,
            total_requests: self.successful_requests.load(Ordering::SeqCst) 
                + self.failed_requests.load(Ordering::SeqCst),
            successful_requests: self.successful_requests.load(Ordering::SeqCst),
            failed_requests: self.failed_requests.load(Ordering::SeqCst),
        })
    }

    /// Spawn a single VU
    async fn spawn_vu(&self, end_tx: mpsc::Sender<()>) {
        let request_id = self.request_counter.fetch_add(1, Ordering::SeqCst);
        self.active_vus.fetch_add(1, Ordering::SeqCst);
        
        // Generate request
        let request = {
            let mut guard = self.requests.lock().await;
            Arc::new(guard.generate_request())
        };
        
        let backend = self.backend.clone();
        let successful = self.successful_requests.clone();
        let failed = self.failed_requests.clone();
        let results = self.results.clone();
        let collector_ref = self.collector_ref.clone();
        
        tokio::spawn(async move {
            let start_time = now_millis();
            let (tx, mut rx) = mpsc::channel(10);
            
            let mut first_token_time: Option<u64> = None;
            let mut inter_token_latencies = Vec::new();
            let mut last_token_time = start_time;
            let mut output_tokens = 0u64;
            let mut success = false;
            let mut error_msg: Option<String> = None;
            
            // Send request
            let req_clone = request.clone();
            let backend_task = tokio::spawn(async move {
                backend.generate(req_clone, tx).await;
            });
            
            // Collect response tokens
            while let Some(response) = rx.recv().await {
                if response.ended {
                    if response.failed {
                        // Response failed
                        error_msg = Some("Request failed".to_string());
                    } else {
                        success = true;
                    }
                    output_tokens = response.num_generated_tokens;
                    break;
                }
                
                // Track token timing from times_to_tokens
                if !response.times_to_tokens.is_empty() {
                    let current_time = now_millis();
                    if first_token_time.is_none() {
                        first_token_time = Some(current_time);
                    } else {
                        inter_token_latencies.push(current_time - last_token_time);
                    }
                    last_token_time = current_time;
                }
                output_tokens = response.num_generated_tokens;
            }
            
            let _ = backend_task.await;
            let end_time = now_millis();
            
            // Record result
            if success {
                successful.fetch_add(1, Ordering::SeqCst);
            } else {
                failed.fetch_add(1, Ordering::SeqCst);
            }
            
            let result = RequestResult {
                request_id: format!("req_{}", request_id),
                success,
                start_time_ms: start_time,
                end_time_ms: end_time,
                time_to_first_token_ms: first_token_time.map(|t| t - start_time),
                inter_token_latencies_ms: inter_token_latencies,
                input_tokens: request.num_prompt_tokens,
                output_tokens,
                error: error_msg,
            };
            
            // Store result locally
            results.write().await.push(result.clone());
            
            // Send to collector if available
            if let Some(collector) = collector_ref {
                if let Ok(msg) = Message::pack(&result) {
                    let _ = collector.send(msg).await;
                }
            }
            
            // Signal VU completion
            let _ = end_tx.send(()).await;
        });
    }

    /// Reset state for new benchmark phase
    fn reset(&mut self) {
        self.request_counter.store(0, Ordering::SeqCst);
        self.active_vus.store(0, Ordering::SeqCst);
        self.successful_requests.store(0, Ordering::SeqCst);
        self.failed_requests.store(0, Ordering::SeqCst);
        self.start_time = None;
        tokio::spawn({
            let results = self.results.clone();
            async move {
                results.write().await.clear();
            }
        });
    }
}

#[async_trait]
impl Actor for ExecutorActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();
        
        if msg_type.ends_with("StartExecutor") {
            let config: StartExecutor = msg.unpack()?;
            self.config = Some(config.clone());
            self.reset();
            
            let result = match config.executor_type.as_str() {
                "constant_vus" => self.run_constant_vus(config).await?,
                "constant_arrival_rate" => self.run_constant_arrival_rate(config).await?,
                _ => return Err(anyhow::anyhow!("Unknown executor type: {}", config.executor_type)),
            };
            
            return Message::pack(&result);
        }
        
        if msg_type.ends_with("StopBenchmark") {
            self.is_running.store(false, Ordering::SeqCst);
            return Message::pack(&"Stopped");
        }
        
        if msg_type.ends_with("GetResults") {
            let results = self.results.read().await.clone();
            return Message::pack(&results);
        }
        
        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}

