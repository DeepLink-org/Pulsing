//! Python bindings for Pulsing Benchmark

use pulsing_bench::{parse_duration, run_benchmark, BenchmarkArgs};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Helper functions for parsing config dict
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

/// Run benchmark
///
/// Args:
///     config: Dictionary with benchmark configuration
///         - url: str (default: "http://localhost:8000")
///         - api_key: str (default: "")
///         - model_name: str (required)
///         - tokenizer_name: str (optional, defaults to model_name)
///         - hf_token: str (optional, for private models)
///         - max_vus: int (default: 128)
///         - duration: str (default: "120s")
///         - warmup: str (default: "30s")
///         - benchmark_kind: str (default: "throughput")
///         - num_rates: int (default: 10)
///         - rates: list[float] (optional)
///         - num_workers: int (default: 4)
///
/// Returns:
///     JSON string with benchmark report
#[pyfunction]
fn benchmark_main<'py>(py: Python<'py>, config: PyObject) -> PyResult<Bound<'py, PyAny>> {
    let config_dict = config.bind(py).downcast::<PyDict>()?;

    // Parse configuration
    let url =
        get_opt_str(config_dict, "url")?.unwrap_or_else(|| "http://localhost:8000".to_string());

    let api_key = get_opt_str(config_dict, "api_key")?.unwrap_or_default();

    // model_name is required
    let model_name = config_dict
        .get_item("model_name")?
        .ok_or_else(|| PyValueError::new_err("model_name is required"))?
        .extract::<String>()?;

    // tokenizer_name is optional, defaults to model_name
    let tokenizer_name = get_opt_str(config_dict, "tokenizer_name")?;

    // hf_token for private models
    let hf_token = get_opt_str(config_dict, "hf_token")?;

    let max_vus = get_opt::<u64>(config_dict, "max_vus")?.unwrap_or(128);

    let duration_str = get_opt_str(config_dict, "duration")?.unwrap_or_else(|| "120s".to_string());
    let duration_secs = parse_duration(&duration_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid duration: {}", e)))?
        .as_secs();

    let warmup_str = get_opt_str(config_dict, "warmup")?.unwrap_or_else(|| "30s".to_string());
    let warmup_secs = parse_duration(&warmup_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid warmup duration: {}", e)))?
        .as_secs();

    let benchmark_kind =
        get_opt_str(config_dict, "benchmark_kind")?.unwrap_or_else(|| "throughput".to_string());

    let num_rates = get_opt::<u64>(config_dict, "num_rates")?.unwrap_or(10);

    let rates: Option<Vec<f64>> = get_opt::<Vec<f64>>(config_dict, "rates")?;

    let num_workers = get_opt::<u32>(config_dict, "num_workers")?.unwrap_or(4);

    let args = BenchmarkArgs {
        url,
        api_key,
        model_name,
        tokenizer_name,
        hf_token,
        max_vus,
        duration_secs,
        warmup_secs,
        benchmark_kind,
        num_rates,
        rates,
        num_workers,
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let report = run_benchmark(args)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Convert report to JSON string for Python
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize report: {}", e)))?;

        Ok(json)
    })
}

/// Pulsing Benchmark Python module
#[pymodule]
fn _bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init()
        .ok();

    m.add_function(wrap_pyfunction!(benchmark_main, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
