//! Python bindings for Pulsing Actor System
//!
//! This crate provides Python bindings for the Pulsing distributed actor framework.
//! It is a standalone module that can be used independently of Dynamo.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod actor;
mod policies;
mod python_executor;

pub use python_executor::{init_python_executor, python_executor, ExecutorError};

/// Benchmark engine type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BenchmarkEngine {
    /// Classic implementation (original)
    #[default]
    Classic,
    /// Actor-based implementation (new)
    Actor,
}

impl BenchmarkEngine {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "classic" | "legacy" | "old" => Some(Self::Classic),
            "actor" | "new" => Some(Self::Actor),
            _ => None,
        }
    }
}

/// Pulsing Actor System Python module
///
/// This module provides:
/// - ActorSystem: Distributed actor system management
/// - Actor types: NodeId, ActorId, ActorRef
/// - Message types: Message, StreamMessage
/// - Streaming: StreamReader, StreamWriter
/// - Load balancing policies: Random, RoundRobin, PowerOfTwo, ConsistentHash, CacheAware
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init()
        .ok();

    // Add actor system classes
    actor::add_to_module(m)?;

    // Add load balancing policies
    policies::add_to_module(m)?;

    // Add benchmark function (supports both classic and actor engines via 'engine' parameter)
    m.add_function(wrap_pyfunction!(benchmark_main, m)?)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// Helper functions for parsing config dict
fn get_opt_str(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<String>().ok()),
        None => Ok(None),
    }
}

fn get_opt<T: for<'a> pyo3::FromPyObject<'a>>(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<T>().ok()),
        None => Ok(None),
    }
}

/// Run benchmark with classic (original) implementation
///
/// Args:
///     config: Dictionary with benchmark configuration
///         - tokenizer_name: str (required)
///         - model_name: str (optional)
///         - max_vus: int (default: 128)
///         - duration: str (default: "120s")
///         - rates: list[float] (optional)
///         - num_rates: int (default: 10)
///         - profile: str (optional)
///         - benchmark_kind: str (default: "sweep")
///         - warmup: str (default: "30s")
///         - url: str (default: "http://localhost:8000")
///         - api_key: str (default: "")
///         - prompt_options: str (optional)
///         - decode_options: str (optional)
///         - dataset: str (default: "hlarcher/inference-benchmarker")
///         - dataset_file: str (default: "share_gpt_filtered_small.json")
///         - extra_meta: str (optional)
///         - run_id: str (optional)
///         - engine: str (optional, "classic" or "actor", default: "classic")
#[pyfunction]
fn benchmark_main<'py>(py: Python<'py>, config: PyObject) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyDict;
    let config_dict = config.bind(py).downcast::<PyDict>()?;
    
    // Check engine parameter
    let engine_str = get_opt_str(config_dict, "engine")?.unwrap_or_else(|| "classic".to_string());
    let engine = BenchmarkEngine::from_str(&engine_str)
        .ok_or_else(|| PyValueError::new_err(format!(
            "Invalid engine '{}'. Use 'classic' or 'actor'", engine_str
        )))?;
    
    match engine {
        BenchmarkEngine::Classic => run_classic_benchmark(py, config_dict),
        BenchmarkEngine::Actor => run_actor_benchmark(py, config_dict),
    }
}

/// Classic benchmark implementation
fn run_classic_benchmark<'py>(
    py: Python<'py>,
    config_dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    use pulsing_bench::BenchmarkArgs;
    use reqwest::Url;

    // Parse configuration from Python dict
    let tokenizer_name = config_dict
        .get_item("tokenizer_name")?
        .ok_or_else(|| PyValueError::new_err("tokenizer_name is required"))?
        .extract::<String>()?;

    let model_name = get_opt_str(config_dict, "model_name")?;

    let max_vus = get_opt::<u64>(config_dict, "max_vus")?.unwrap_or(128);

    let duration_str = get_opt_str(config_dict, "duration")?.unwrap_or_else(|| "120s".to_string());
    let duration = pulsing_bench::parse_duration(&duration_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid duration: {}", e)))?;

    let rates: Option<Vec<f64>> = get_opt::<Vec<f64>>(config_dict, "rates")?;

    let num_rates = get_opt::<u64>(config_dict, "num_rates")?.unwrap_or(10);

    let profile = get_opt_str(config_dict, "profile")?;

    let benchmark_kind =
        get_opt_str(config_dict, "benchmark_kind")?.unwrap_or_else(|| "sweep".to_string());

    let warmup_str = get_opt_str(config_dict, "warmup")?.unwrap_or_else(|| "30s".to_string());
    let warmup = pulsing_bench::parse_duration(&warmup_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid warmup duration: {}", e)))?;

    let url_str =
        get_opt_str(config_dict, "url")?.unwrap_or_else(|| "http://localhost:8000".to_string());
    let url = url_str
        .parse::<Url>()
        .map_err(|e| PyValueError::new_err(format!("Invalid URL: {}", e)))?;

    let api_key = get_opt_str(config_dict, "api_key")?.unwrap_or_default();

    let prompt_options_str = get_opt_str(config_dict, "prompt_options")?;
    let prompt_options = if let Some(s) = prompt_options_str {
        Some(
            pulsing_bench::parse_tokenizer_options(&s)
                .map_err(|e| PyValueError::new_err(format!("Invalid prompt_options: {}", e)))?,
        )
    } else {
        None
    };

    let decode_options_str = get_opt_str(config_dict, "decode_options")?;
    let decode_options = if let Some(s) = decode_options_str {
        Some(
            pulsing_bench::parse_tokenizer_options(&s)
                .map_err(|e| PyValueError::new_err(format!("Invalid decode_options: {}", e)))?,
        )
    } else {
        None
    };

    let dataset = get_opt_str(config_dict, "dataset")?
        .unwrap_or_else(|| "hlarcher/inference-benchmarker".to_string());

    let dataset_file = get_opt_str(config_dict, "dataset_file")?
        .unwrap_or_else(|| "share_gpt_filtered_small.json".to_string());

    let extra_meta_str = get_opt_str(config_dict, "extra_meta")?;
    let extra_meta = if let Some(s) = extra_meta_str {
        Some(
            pulsing_bench::parse_key_val(&s)
                .map_err(|e| PyValueError::new_err(format!("Invalid extra_meta: {}", e)))?,
        )
    } else {
        None
    };

    let run_id = get_opt_str(config_dict, "run_id")?;

    let args = BenchmarkArgs {
        tokenizer_name,
        model_name,
        max_vus,
        duration,
        rates,
        num_rates,
        profile,
        benchmark_kind,
        warmup,
        url,
        api_key,
        prompt_options,
        decode_options,
        dataset,
        dataset_file,
        extra_meta,
        run_id,
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        pulsing_bench::benchmark_main_async(args)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

/// Actor-based benchmark implementation
fn run_actor_benchmark<'py>(
    py: Python<'py>,
    config_dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    use pulsing_bench::ActorBenchmarkArgs;

    // Parse configuration - simpler than classic, no tokenizer required
    let url = get_opt_str(config_dict, "url")?
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    
    let api_key = get_opt_str(config_dict, "api_key")?.unwrap_or_default();
    
    // model_name is required for actor mode
    let model_name = config_dict
        .get_item("model_name")?
        .or_else(|| config_dict.get_item("tokenizer_name").ok().flatten())
        .ok_or_else(|| PyValueError::new_err("model_name is required for actor engine"))?
        .extract::<String>()?;

    let max_vus = get_opt::<u64>(config_dict, "max_vus")?.unwrap_or(128);

    let duration_str = get_opt_str(config_dict, "duration")?.unwrap_or_else(|| "120s".to_string());
    let duration_secs = pulsing_bench::parse_duration(&duration_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid duration: {}", e)))?
        .as_secs();

    let warmup_str = get_opt_str(config_dict, "warmup")?.unwrap_or_else(|| "30s".to_string());
    let warmup_secs = pulsing_bench::parse_duration(&warmup_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid warmup duration: {}", e)))?
        .as_secs();

    let benchmark_kind =
        get_opt_str(config_dict, "benchmark_kind")?.unwrap_or_else(|| "throughput".to_string());

    let num_rates = get_opt::<u64>(config_dict, "num_rates")?.unwrap_or(10);

    let rates: Option<Vec<f64>> = get_opt::<Vec<f64>>(config_dict, "rates")?;

    let num_workers = get_opt::<u32>(config_dict, "num_workers")?.unwrap_or(4);

    let args = ActorBenchmarkArgs {
        url,
        api_key,
        model_name,
        max_vus,
        duration_secs,
        warmup_secs,
        benchmark_kind,
        num_rates,
        rates,
        num_workers,
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let report = pulsing_bench::run_actor_benchmark(args)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Convert report to JSON string for Python
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize report: {}", e)))?;
        
        Ok(json)
    })
}
