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

    // Add benchmark function
    m.add_function(wrap_pyfunction!(benchmark_main, m)?)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

#[pyfunction]
fn benchmark_main<'py>(py: Python<'py>, config: PyObject) -> PyResult<Bound<'py, PyAny>> {
    use pulsing_bench::BenchmarkArgs;
    use reqwest::Url;

    use pyo3::types::PyDict;
    let config_dict = config.bind(py).downcast::<PyDict>()?;

    // Helper function to get optional value from dict
    fn get_opt_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
        match dict.get_item(key)? {
            Some(v) => Ok(v.extract::<String>().ok()),
            None => Ok(None),
        }
    }

    fn get_opt<T: for<'a> pyo3::FromPyObject<'a>>(
        dict: &Bound<'_, PyDict>,
        key: &str,
    ) -> PyResult<Option<T>> {
        match dict.get_item(key)? {
            Some(v) => Ok(v.extract::<T>().ok()),
            None => Ok(None),
        }
    }

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
