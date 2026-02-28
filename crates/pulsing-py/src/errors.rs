//! Python exception bindings for Pulsing errors
//!
//! This module converts Rust error types directly to Python exception instances
//! by importing `pulsing.exceptions` and calling the appropriate class constructor.
//! No intermediate JSON/string encoding is needed.

use pulsing_actor::error::{ActorError, PulsingError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::OnceLock;

static EXCEPTIONS_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();

fn get_exceptions(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    if let Some(m) = EXCEPTIONS_MODULE.get() {
        return Ok(m.bind(py).clone());
    }
    let module = py.import("pulsing.exceptions")?;
    // Another thread may have set it concurrently; that's fine — both see the same module.
    let _ = EXCEPTIONS_MODULE.set(module.clone().unbind());
    Ok(module)
}

fn try_create_py_err(py: Python<'_>, err: &PulsingError) -> PyResult<PyErr> {
    let exc = get_exceptions(py)?;
    match err {
        PulsingError::Runtime(re) => {
            let cls = exc.getattr("PulsingRuntimeError")?;
            Ok(PyErr::from_value(cls.call1((re.to_string(),))?))
        }
        PulsingError::Actor(ae) => match ae {
            ActorError::Business {
                code,
                message,
                details,
            } => {
                let cls = exc.getattr("PulsingBusinessError")?;
                let exc_obj = if let Some(d) = details {
                    cls.call1((*code, message.as_str(), d.as_str()))?
                } else {
                    cls.call1((*code, message.as_str()))?
                };
                Ok(PyErr::from_value(exc_obj))
            }
            ActorError::System { error, recoverable } => {
                let cls = exc.getattr("PulsingSystemError")?;
                Ok(PyErr::from_value(
                    cls.call1((error.as_str(), *recoverable))?,
                ))
            }
            ActorError::Timeout {
                operation,
                duration_ms,
            } => {
                let cls = exc.getattr("PulsingTimeoutError")?;
                Ok(PyErr::from_value(
                    cls.call1((operation.as_str(), *duration_ms))?,
                ))
            }
            ActorError::Unsupported { operation } => {
                let cls = exc.getattr("PulsingUnsupportedError")?;
                Ok(PyErr::from_value(cls.call1((operation.as_str(),))?))
            }
        },
    }
}

/// Convert Rust PulsingError to Python exception.
///
/// Directly instantiates the matching `pulsing.exceptions` class so the caller
/// receives a typed exception (PulsingRuntimeError, PulsingBusinessError, …)
/// without any intermediate JSON envelope or string parsing.
pub fn pulsing_error_to_py_err(err: PulsingError) -> PyErr {
    Python::with_gil(|py| {
        try_create_py_err(py, &err).unwrap_or_else(|_| PyRuntimeError::new_err(err.to_string()))
    })
}

/// Add error classes to Python module.
///
/// Exception classes are defined in Python (`pulsing/exceptions.py`); nothing
/// to register here.
pub fn add_to_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
