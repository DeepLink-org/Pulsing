//! Python exception bindings for Pulsing errors
//!
//! This module converts Rust error types to Python exceptions.
//! Due to PyO3 abi3 limitations, we use PyRuntimeError as the base
//! and let Python layer re-raise as appropriate exception types.

use pulsing_actor::error::{PulsingError, RuntimeError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert Rust PulsingError to appropriate Python exception
///
/// This function prefixes error messages with error type markers so Python
/// layer can identify and re-raise as appropriate exception types.
pub fn pulsing_error_to_py_err(err: PulsingError) -> PyErr {
    let err_msg = err.to_string();

    match &err {
        // Actor errors (user code errors) -> prefix with "ACTOR_ERROR:"
        PulsingError::Actor(_actor_err) => {
            PyRuntimeError::new_err(format!("ACTOR_ERROR:{}", err_msg))
        }
        // Runtime errors (framework errors) -> prefix with "RUNTIME_ERROR:"
        PulsingError::Runtime(runtime_err) => {
            // Extract actor name if available for runtime errors
            let actor_name = match runtime_err {
                RuntimeError::ActorNotFound { name } => Some(name.clone()),
                RuntimeError::ActorAlreadyExists { name } => Some(name.clone()),
                RuntimeError::ActorNotLocal { name } => Some(name.clone()),
                RuntimeError::ActorStopped { name } => Some(name.clone()),
                RuntimeError::ActorMailboxFull { name } => Some(name.clone()),
                RuntimeError::InvalidActorPath { path: _ } => None,
                RuntimeError::MessageTypeMismatch { .. } => None,
                RuntimeError::ActorSpawnFailed { .. } => None,
                _ => None,
            };

            let full_msg = if let Some(ref name) = actor_name {
                format!("RUNTIME_ERROR:{}:actor={}", err_msg, name)
            } else {
                format!("RUNTIME_ERROR:{}", err_msg)
            };

            PyRuntimeError::new_err(full_msg)
        }
    }
}

/// Convert PulsingError to Python exception (preferred method)
pub fn pulsing_error_to_py_err_direct(err: PulsingError) -> PyErr {
    pulsing_error_to_py_err(err)
}

/// Add error classes to Python module
///
/// Note: In abi3 mode, we can't create custom exception classes directly.
/// Exception classes are defined in Python (pulsing/exceptions.py).
/// This function is kept for API consistency.
pub fn add_to_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Error classes are defined in Python layer
    Ok(())
}
