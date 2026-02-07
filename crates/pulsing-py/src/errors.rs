//! Python exception bindings for Pulsing errors
//!
//! This module converts Rust error types to Python exceptions using
//! JSON-structured error envelopes instead of string prefixes.
//!
//! The JSON envelope format:
//! ```json
//! {
//!   "category": "actor" | "runtime",
//!   // For actor errors (category="actor"):
//!   "error": { "type": "business", "code": 400, ... },
//!   // For runtime errors (category="runtime"):
//!   "kind": "actor_not_found",
//!   "message": "Actor not found: my-actor",
//!   "actor_name": "my-actor"  // optional
//! }
//! ```

use pulsing_actor::error::PulsingError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// JSON marker prefix for structured error envelopes.
/// Python layer detects this prefix and parses the JSON payload.
pub const ERROR_ENVELOPE_PREFIX: &str = "__PULSING_ERROR__:";

/// Convert Rust PulsingError to Python exception using JSON envelope.
///
/// Instead of string-prefix-based encoding (fragile, requires regex parsing),
/// this uses a JSON-structured envelope that Python can reliably decode.
pub fn pulsing_error_to_py_err(err: PulsingError) -> PyErr {
    let json_str = match &err {
        PulsingError::Actor(actor_err) => {
            // ActorError already derives Serialize with serde(tag = "type")
            let actor_json = serde_json::to_value(actor_err).unwrap_or_else(|_| {
                serde_json::json!({"type": "system", "error": err.to_string(), "recoverable": true})
            });
            serde_json::json!({
                "category": "actor",
                "error": actor_json,
            })
            .to_string()
        }
        PulsingError::Runtime(runtime_err) => serde_json::json!({
            "category": "runtime",
            "kind": runtime_err.kind(),
            "message": runtime_err.to_string(),
            "actor_name": runtime_err.actor_name(),
        })
        .to_string(),
    };

    PyRuntimeError::new_err(format!("{}{}", ERROR_ENVELOPE_PREFIX, json_str))
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

#[cfg(test)]
mod tests {
    use super::*;
    use pulsing_actor::error::{ActorError, RuntimeError};

    #[test]
    fn test_actor_error_envelope() {
        let err = PulsingError::Actor(ActorError::business(400, "Invalid input", None));
        let py_err = pulsing_error_to_py_err(err);
        let msg = py_err.to_string();
        assert!(msg.starts_with(ERROR_ENVELOPE_PREFIX));

        let json_str = &msg[ERROR_ENVELOPE_PREFIX.len()..];
        let envelope: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(envelope["category"], "actor");
        assert_eq!(envelope["error"]["type"], "business");
        assert_eq!(envelope["error"]["code"], 400);
    }

    #[test]
    fn test_runtime_error_envelope() {
        let err = PulsingError::Runtime(RuntimeError::actor_not_found("my-actor"));
        let py_err = pulsing_error_to_py_err(err);
        let msg = py_err.to_string();
        assert!(msg.starts_with(ERROR_ENVELOPE_PREFIX));

        let json_str = &msg[ERROR_ENVELOPE_PREFIX.len()..];
        let envelope: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(envelope["category"], "runtime");
        assert_eq!(envelope["kind"], "actor_not_found");
        assert_eq!(envelope["actor_name"], "my-actor");
    }

    #[test]
    fn test_anyhow_error_conversion() {
        let anyhow_err = anyhow::anyhow!("something went wrong");
        let py_err = pulsing_error_to_py_err(PulsingError::from(RuntimeError::Other(
            anyhow_err.to_string(),
        )));
        let msg = py_err.to_string();
        assert!(msg.starts_with(ERROR_ENVELOPE_PREFIX));

        let json_str = &msg[ERROR_ENVELOPE_PREFIX.len()..];
        let envelope: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(envelope["category"], "runtime");
        assert_eq!(envelope["kind"], "other");
    }
}
