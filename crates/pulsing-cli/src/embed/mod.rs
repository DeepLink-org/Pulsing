//! RustPython embedding for extension-mode workflows.

mod python;

pub use python::{delegate_to_python_cli, extension_mode_available, run_workflow_script};
