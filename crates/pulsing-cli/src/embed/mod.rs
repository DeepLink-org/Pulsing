//! RustPython (`rustpython_vm`) — in-process interpreter for Path B.

mod python;

pub use python::{delegate_to_python_cli, run_python_script};
