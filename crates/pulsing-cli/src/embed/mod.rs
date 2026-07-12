//! RustPython embedding for extension-mode workflows.

mod python;

pub use python::{delegate_to_python_cli, extension_mode_available, run_workflow_script};

#[allow(dead_code)]
pub fn warn_extension_mode() {
    eprintln!("{}", crate::help::EXTENSION_MODE_HINT);
}

pub fn warn_legacy_mode() {
    eprintln!("{}", crate::help::LEGACY_HINT);
}
