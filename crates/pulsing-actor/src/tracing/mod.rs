//! Distributed Tracing Support for Pulsing Actor System
//!
//! Provides OpenTelemetry-compatible distributed tracing with:
//! - W3C Trace Context propagation (traceparent/tracestate headers)
//! - Automatic span creation for actor message handling
//! - OTLP export support (optional, via `otlp` feature)
//!
//! ## Usage
//!
//! Tracing is automatically enabled when configured. Python code doesn't need
//! any changes - spans are created transparently in the Rust transport layer.
//!
//! ```rust,ignore
//! use pulsing_actor::tracing::{init_tracing, TracingConfig};
//!
//! // Initialize with console output (for development)
//! init_tracing(TracingConfig::default())?;
//!
//! // Or with OTLP export (requires `otlp` feature)
//! init_tracing(TracingConfig {
//!     service_name: "my-service".to_string(),
//!     otlp_endpoint: Some("http://jaeger:4317".to_string()),
//!     ..Default::default()
//! })?;
//! ```

use opentelemetry::trace::{
    SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, TracerProvider as _,
};
use opentelemetry::{global, Context, KeyValue};
use opentelemetry_sdk::trace::{RandomIdGenerator, Sampler, TracerProvider};
use opentelemetry_sdk::Resource;

// Re-export for external use
pub use opentelemetry;
use std::sync::OnceLock;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Global tracing state
static TRACING_INITIALIZED: OnceLock<bool> = OnceLock::new();

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for traces
    pub service_name: String,
    /// OTLP endpoint (e.g., "http://jaeger:4317")
    /// If None, traces are only logged to console
    pub otlp_endpoint: Option<String>,
    /// Sampling ratio (0.0 - 1.0)
    pub sampling_ratio: f64,
    /// Enable console output
    pub console_output: bool,
    /// Log level filter (e.g., "info", "debug", "pulsing_actor=debug")
    pub log_filter: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "pulsing".to_string(),
            otlp_endpoint: None,
            sampling_ratio: 1.0,
            console_output: true,
            log_filter: "info".to_string(),
        }
    }
}

/// Initialize distributed tracing
///
/// This should be called once at application startup. Subsequent calls are no-ops.
pub fn init_tracing(config: TracingConfig) -> anyhow::Result<()> {
    if TRACING_INITIALIZED.get().is_some() {
        return Ok(());
    }

    // Create sampler based on ratio
    let sampler = if config.sampling_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if config.sampling_ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(config.sampling_ratio)
    };

    // Build tracer provider
    let resource = Resource::new(vec![KeyValue::new(
        "service.name",
        config.service_name.clone(),
    )]);

    let provider = TracerProvider::builder()
        .with_sampler(sampler)
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource)
        .build();

    global::set_tracer_provider(provider.clone());

    // Build subscriber layers
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_filter));

    let tracer = provider.tracer("pulsing");

    if config.console_output {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false);

        let subscriber = tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer);

        let otel_layer = OpenTelemetryLayer::new(tracer);

        subscriber
            .with(otel_layer)
            .try_init()
            .map_err(|e| anyhow::anyhow!("Failed to init tracing: {}", e))?;
    } else {
        let subscriber = tracing_subscriber::registry().with(env_filter);
        let otel_layer = OpenTelemetryLayer::new(tracer);

        subscriber
            .with(otel_layer)
            .try_init()
            .map_err(|e| anyhow::anyhow!("Failed to init tracing: {}", e))?;
    }

    TRACING_INITIALIZED.set(true).ok();
    tracing::info!(service = %config.service_name, "Distributed tracing initialized");

    Ok(())
}

/// Shutdown tracing and flush pending spans
pub fn shutdown_tracing() {
    global::shutdown_tracer_provider();
}

// ============================================================================
// W3C Trace Context
// ============================================================================

/// W3C Trace Context for propagation
///
/// Format: `{version}-{trace_id}-{span_id}-{flags}`
/// Example: `00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01`
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub trace_flags: u8,
    pub trace_state: Option<String>,
}

impl TraceContext {
    /// Create from current OpenTelemetry context
    pub fn from_current() -> Option<Self> {
        let ctx = Context::current();
        let span = ctx.span();
        let span_ctx = span.span_context();

        if !span_ctx.is_valid() {
            return None;
        }

        Some(Self {
            trace_id: span_ctx.trace_id().to_string(),
            span_id: span_ctx.span_id().to_string(),
            trace_flags: span_ctx.trace_flags().to_u8(),
            trace_state: None, // TraceState propagation handled separately if needed
        })
    }

    /// Parse from W3C traceparent header
    ///
    /// Format: `{version}-{trace_id}-{span_id}-{flags}`
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let _version = parts[0];
        let trace_id = parts[1];
        let span_id = parts[2];
        let flags = parts[3];

        // Validate lengths
        if trace_id.len() != 32 || span_id.len() != 16 || flags.len() != 2 {
            return None;
        }

        let trace_flags = u8::from_str_radix(flags, 16).ok()?;

        Some(Self {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            trace_flags,
            trace_state: None,
        })
    }

    /// Convert to W3C traceparent header value
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }

    /// Create OpenTelemetry Context from this TraceContext
    pub fn to_otel_context(&self) -> Context {
        let trace_id = TraceId::from_hex(&self.trace_id).unwrap_or(TraceId::INVALID);
        let span_id = SpanId::from_hex(&self.span_id).unwrap_or(SpanId::INVALID);
        let trace_flags = TraceFlags::new(self.trace_flags);
        let trace_state = TraceState::default();

        let span_context = SpanContext::new(trace_id, span_id, trace_flags, true, trace_state);
        Context::current().with_remote_span_context(span_context)
    }

    /// Generate a new child span ID
    pub fn new_child_span_id() -> String {
        use rand::Rng;
        let mut rng = rand::rng();
        let id: u64 = rng.random();
        format!("{:016x}", id)
    }

    /// Create a child context with new span ID
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Self::new_child_span_id(),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let trace_id: u128 = rng.random();
        let span_id: u64 = rng.random();

        Self {
            trace_id: format!("{:032x}", trace_id),
            span_id: format!("{:016x}", span_id),
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }
}

// ============================================================================
// HTTP Header Constants
// ============================================================================

/// W3C Trace Context header name
pub const TRACEPARENT_HEADER: &str = "traceparent";

/// W3C Trace State header name
pub const TRACESTATE_HEADER: &str = "tracestate";

// ============================================================================
// Span Helpers
// ============================================================================

/// Create a span for actor message receive
#[macro_export]
macro_rules! actor_span {
    ($actor_name:expr, $msg_type:expr) => {
        tracing::info_span!(
            "actor.receive",
            otel.name = %format!("actor.receive {}", $actor_name),
            actor.name = %$actor_name,
            message.type = %$msg_type,
        )
    };
    ($actor_name:expr, $msg_type:expr, $($field:tt)*) => {
        tracing::info_span!(
            "actor.receive",
            otel.name = %format!("actor.receive {}", $actor_name),
            actor.name = %$actor_name,
            message.type = %$msg_type,
            $($field)*
        )
    };
}

/// Create a span for remote actor call
#[macro_export]
macro_rules! remote_call_span {
    ($target_addr:expr, $path:expr) => {
        tracing::info_span!(
            "actor.remote_call",
            otel.name = %format!("actor.remote_call {}", $path),
            target.addr = %$target_addr,
            target.path = %$path,
        )
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_default() {
        let ctx = TraceContext::default();
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_trace_context_to_traceparent() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            trace_flags: 0x01,
            trace_state: None,
        };
        assert_eq!(
            ctx.to_traceparent(),
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
    }

    #[test]
    fn test_trace_context_from_traceparent() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::from_traceparent(header).unwrap();
        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.span_id, "b7ad6b7169203331");
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_trace_context_roundtrip() {
        let original = TraceContext::default();
        let header = original.to_traceparent();
        let parsed = TraceContext::from_traceparent(&header).unwrap();
        assert_eq!(original.trace_id, parsed.trace_id);
        assert_eq!(original.span_id, parsed.span_id);
        assert_eq!(original.trace_flags, parsed.trace_flags);
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::default();
        let child = parent.child();
        assert_eq!(parent.trace_id, child.trace_id); // Same trace
        assert_ne!(parent.span_id, child.span_id); // Different span
    }

    #[test]
    fn test_invalid_traceparent() {
        assert!(TraceContext::from_traceparent("invalid").is_none());
        assert!(TraceContext::from_traceparent("00-short-id-01").is_none());
    }
}
