//! Rust-side OpenTelemetry / `tracing` init for Python processes.

use pulsing_actor::tracing::{init_tracing, shutdown_tracing, TracingConfig};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (service_name=None, console_output=None))]
fn init_distributed_tracing(
    service_name: Option<String>,
    console_output: Option<bool>,
) -> PyResult<()> {
    let mut cfg = TracingConfig::default();
    if let Some(s) = service_name {
        cfg.service_name = s;
    }
    if let Some(c) = console_output {
        cfg.console_output = c;
    }
    init_tracing(cfg).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    Ok(())
}

#[pyfunction]
fn shutdown_distributed_tracing() {
    shutdown_tracing();
}

pub(crate) fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_distributed_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_distributed_tracing, m)?)?;
    Ok(())
}
