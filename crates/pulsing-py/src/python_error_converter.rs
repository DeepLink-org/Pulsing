//! Convert Python exceptions to Rust ActorError
//!
//! Uses `isinstance` checks against `pulsing.exceptions` classes — matching
//! most-specific types first — then falls back to standard Python exception
//! types for interoperability.

use pulsing_actor::error::ActorError;
use pyo3::exceptions::{PyTimeoutError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Convert Python exception to ActorError.
///
/// Matching order (most-specific first):
/// 1. Pulsing custom types: Business → System → Timeout → Unsupported
/// 2. Standard Python types: TimeoutError → ValueError/TypeError → fallback
pub fn convert_python_exception_to_actor_error(
    py: Python,
    err: &PyErr,
) -> anyhow::Result<ActorError> {
    let err_obj = err.value(py);

    if let Ok(exc) = py.import("pulsing.exceptions") {
        // PulsingBusinessError — check before PulsingActorError (more specific)
        if let Ok(cls) = exc.getattr("PulsingBusinessError") {
            if err_obj.is_instance(&cls).unwrap_or(false) {
                let code = err_obj.getattr("code")?.extract::<u32>()?;
                let message = err_obj.getattr("message")?.extract::<String>()?;
                let details = err_obj
                    .getattr("details")
                    .ok()
                    .and_then(|d| d.extract::<Option<String>>().ok())
                    .flatten();
                return Ok(ActorError::business(code, message, details));
            }
        }

        if let Ok(cls) = exc.getattr("PulsingSystemError") {
            if err_obj.is_instance(&cls).unwrap_or(false) {
                let error = err_obj.getattr("error")?.extract::<String>()?;
                let recoverable = err_obj
                    .getattr("recoverable")
                    .ok()
                    .and_then(|r| r.extract::<bool>().ok())
                    .unwrap_or(true);
                return Ok(ActorError::system(error, recoverable));
            }
        }

        if let Ok(cls) = exc.getattr("PulsingTimeoutError") {
            if err_obj.is_instance(&cls).unwrap_or(false) {
                let operation = err_obj.getattr("operation")?.extract::<String>()?;
                let duration_ms = err_obj
                    .getattr("duration_ms")
                    .ok()
                    .and_then(|d| d.extract::<u64>().ok())
                    .unwrap_or(0);
                return Ok(ActorError::timeout(operation, duration_ms));
            }
        }

        if let Ok(cls) = exc.getattr("PulsingUnsupportedError") {
            if err_obj.is_instance(&cls).unwrap_or(false) {
                let operation = err_obj.getattr("operation")?.extract::<String>()?;
                return Ok(ActorError::unsupported(operation));
            }
        }
    }

    // Standard Python exception fallback
    if err.is_instance_of::<PyTimeoutError>(py) {
        return Ok(ActorError::timeout("python_operation", 0));
    }

    if err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py) {
        return Ok(ActorError::business(400, err.to_string(), None));
    }

    Ok(ActorError::system(err.to_string(), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_timeout_error() {
        Python::with_gil(|py| {
            let err = PyTimeoutError::new_err("Operation timed out");
            let actor_err = convert_python_exception_to_actor_error(py, &err).unwrap();
            assert!(matches!(actor_err, ActorError::Timeout { .. }));
        });
    }

    #[test]
    fn test_convert_value_error() {
        Python::with_gil(|py| {
            let err = PyValueError::new_err("Invalid value");
            let actor_err = convert_python_exception_to_actor_error(py, &err).unwrap();
            assert!(matches!(actor_err, ActorError::Business { code: 400, .. }));
        });
    }
}
