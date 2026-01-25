//! Convert Python exceptions to Rust ActorError
//!
//! This module provides automatic conversion from Python exceptions
//! to unified ActorError types, enabling seamless error handling
//! across Rust and Python boundaries.

use pulsing_actor::error::ActorError;
use pyo3::exceptions::{PyTimeoutError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Convert Python exception (PyErr) to ActorError
///
/// This function automatically classifies Python exceptions:
/// - ValueError, TypeError -> Business error
/// - TimeoutError -> Timeout error
/// - Other exceptions -> System error
pub fn convert_python_exception_to_actor_error(
    py: Python,
    err: &PyErr,
) -> anyhow::Result<ActorError> {
    // Try to extract exception type and message
    let err_type = err.get_type(py);
    let type_name = err_type.name()?.to_string();
    let err_msg = err.to_string();

    // Check for specific exception types
    if err.is_instance_of::<PyTimeoutError>(py) {
        // Timeout error
        return Ok(ActorError::timeout("python_operation", 0));
    }

    if err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py) {
        // Business error: validation/type errors
        return Ok(ActorError::business(400, err_msg, None));
    }

    // Check if it's a custom Pulsing exception
    // Try to extract error details from exception attributes
    let py_err_obj = err.value(py);

    // Check for PulsingBusinessError
    if let Ok(code_attr) = py_err_obj.getattr("code") {
        if let Ok(code) = code_attr.extract::<u32>() {
            let message_attr = py_err_obj.getattr("message").ok();
            let message = message_attr
                .and_then(|m| m.extract::<String>().ok())
                .unwrap_or_else(|| err_msg.clone());

            let details_attr = py_err_obj.getattr("details").ok();
            let details = details_attr.and_then(|d| d.extract::<String>().ok());

            return Ok(ActorError::business(code, message, details));
        }
    }

    // Check for PulsingSystemError
    if let Ok(error_attr) = py_err_obj.getattr("error") {
        if let Ok(error_msg) = error_attr.extract::<String>() {
            let recoverable_attr = py_err_obj.getattr("recoverable").ok();
            let recoverable = recoverable_attr
                .and_then(|r| r.extract::<bool>().ok())
                .unwrap_or(true);

            return Ok(ActorError::system(error_msg, recoverable));
        }
    }

    // Check for PulsingTimeoutError (has both operation and duration_ms)
    if let Ok(operation_attr) = py_err_obj.getattr("operation") {
        if let Ok(operation) = operation_attr.extract::<String>() {
            let duration_attr = py_err_obj.getattr("duration_ms").ok();
            if let Some(duration_ms) = duration_attr.and_then(|d| d.extract::<u64>().ok()) {
                // Has duration_ms -> Timeout error
                return Ok(ActorError::timeout(operation, duration_ms));
            }
        }
    }

    // Check for PulsingUnsupportedError (by type name or operation attribute without duration_ms)
    if type_name.contains("Unsupported") || type_name.contains("unsupported") {
        if let Ok(operation_attr) = py_err_obj.getattr("operation") {
            if let Ok(operation) = operation_attr.extract::<String>() {
                return Ok(ActorError::unsupported(operation));
            }
        }
        // Fallback: use error message as operation
        return Ok(ActorError::unsupported(err_msg));
    }

    // Default: classify based on exception type name
    match type_name.as_str() {
        "TimeoutError" | "asyncio.TimeoutError" => Ok(ActorError::timeout("python_operation", 0)),
        "ValueError" | "TypeError" | "KeyError" | "AttributeError" => {
            // Business errors: user input errors
            Ok(ActorError::business(400, err_msg, None))
        }
        "RuntimeError" | "SystemError" | "OSError" | "IOError" => {
            // System errors: internal errors
            Ok(ActorError::system(err_msg, true))
        }
        _ => {
            // Unknown exception type: treat as system error
            Ok(ActorError::system(
                format!("{}: {}", type_name, err_msg),
                true,
            ))
        }
    }
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
