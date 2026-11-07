//! Python bindings for load balancing policies
//!
//! This module provides Python bindings for the load balancing policies
//! that can be used with the router actor.

use pulsing_actor::policies::{
    ConsistentHashPolicy, LoadBalancingPolicy, PowerOfTwoPolicy, RandomPolicy, RoundRobinPolicy,
    Worker,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// PyWorker - Python-side worker representation
// ============================================================================

/// Worker info that Python can manipulate
#[pyclass(name = "WorkerInfo")]
#[derive(Debug, Clone)]
pub struct PyWorkerInfo {
    url: String,
    healthy: Arc<AtomicBool>,
    load: Arc<AtomicUsize>,
    processed: Arc<AtomicU64>,
}

#[pymethods]
impl PyWorkerInfo {
    #[new]
    fn new(url: String) -> Self {
        Self {
            url,
            healthy: Arc::new(AtomicBool::new(true)),
            load: Arc::new(AtomicUsize::new(0)),
            processed: Arc::new(AtomicU64::new(0)),
        }
    }

    #[getter]
    fn url(&self) -> &str {
        &self.url
    }

    #[getter]
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    #[setter]
    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }

    #[getter]
    fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
    }

    #[getter]
    fn processed(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    fn increment_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerInfo(url='{}', healthy={}, load={})",
            self.url,
            self.is_healthy(),
            self.load()
        )
    }
}

// Implement Worker trait for PyWorkerInfo wrapper
#[derive(Debug)]
struct PyWorkerWrapper {
    inner: PyWorkerInfo,
}

impl Worker for PyWorkerWrapper {
    fn url(&self) -> &str {
        &self.inner.url
    }

    fn is_healthy(&self) -> bool {
        self.inner.healthy.load(Ordering::Relaxed)
    }

    fn set_healthy(&mut self, healthy: bool) {
        self.inner.healthy.store(healthy, Ordering::Relaxed);
    }

    fn load(&self) -> usize {
        self.inner.load.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.inner.load.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        self.inner.load.fetch_sub(1, Ordering::Relaxed);
    }

    fn increment_processed(&self) {
        self.inner.processed.fetch_add(1, Ordering::Relaxed);
    }

    fn processed(&self) -> u64 {
        self.inner.processed.load(Ordering::Relaxed)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Helper to convert Python worker list to Rust
fn py_workers_to_rust(workers: &[PyWorkerInfo]) -> Vec<Arc<dyn Worker>> {
    workers
        .iter()
        .map(|w| Arc::new(PyWorkerWrapper { inner: w.clone() }) as Arc<dyn Worker>)
        .collect()
}

// ============================================================================
// Policy Wrappers
// ============================================================================

/// Random load balancing policy
#[pyclass(name = "RandomPolicy")]
pub struct PyRandomPolicy {
    inner: RandomPolicy,
}

#[pymethods]
impl PyRandomPolicy {
    #[new]
    fn new() -> Self {
        Self {
            inner: RandomPolicy::new(),
        }
    }

    /// Select a worker index from the list of workers
    #[pyo3(signature = (workers, request_text=None))]
    fn select_worker(
        &self,
        workers: Vec<PyWorkerInfo>,
        request_text: Option<&str>,
    ) -> Option<usize> {
        let rust_workers = py_workers_to_rust(&workers);
        self.inner.select_worker(&rust_workers, request_text)
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        "RandomPolicy()".to_string()
    }
}

/// Round-robin load balancing policy
#[pyclass(name = "RoundRobinPolicy")]
pub struct PyRoundRobinPolicy {
    inner: RoundRobinPolicy,
}

#[pymethods]
impl PyRoundRobinPolicy {
    #[new]
    fn new() -> Self {
        Self {
            inner: RoundRobinPolicy::new(),
        }
    }

    #[pyo3(signature = (workers, request_text=None))]
    fn select_worker(
        &self,
        workers: Vec<PyWorkerInfo>,
        request_text: Option<&str>,
    ) -> Option<usize> {
        let rust_workers = py_workers_to_rust(&workers);
        self.inner.select_worker(&rust_workers, request_text)
    }

    fn reset(&self) {
        self.inner.reset();
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        "RoundRobinPolicy()".to_string()
    }
}

/// Power-of-two choices load balancing policy
#[pyclass(name = "PowerOfTwoPolicy")]
pub struct PyPowerOfTwoPolicy {
    inner: PowerOfTwoPolicy,
}

#[pymethods]
impl PyPowerOfTwoPolicy {
    #[new]
    fn new() -> Self {
        Self {
            inner: PowerOfTwoPolicy::new(),
        }
    }

    #[pyo3(signature = (workers, request_text=None))]
    fn select_worker(
        &self,
        workers: Vec<PyWorkerInfo>,
        request_text: Option<&str>,
    ) -> Option<usize> {
        let rust_workers = py_workers_to_rust(&workers);
        self.inner.select_worker(&rust_workers, request_text)
    }

    /// Update cached load information
    fn update_loads(&self, loads: HashMap<String, isize>) {
        self.inner.update_loads(&loads);
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        "PowerOfTwoPolicy()".to_string()
    }
}

/// Consistent hash load balancing policy
#[pyclass(name = "ConsistentHashPolicy")]
pub struct PyConsistentHashPolicy {
    inner: ConsistentHashPolicy,
}

#[pymethods]
impl PyConsistentHashPolicy {
    #[new]
    fn new() -> Self {
        Self {
            inner: ConsistentHashPolicy::new(),
        }
    }

    #[pyo3(signature = (workers, request_text=None, headers=None))]
    fn select_worker(
        &self,
        workers: Vec<PyWorkerInfo>,
        request_text: Option<&str>,
        headers: Option<HashMap<String, String>>,
    ) -> Option<usize> {
        let rust_workers = py_workers_to_rust(&workers);
        self.inner
            .select_worker_with_headers(&rust_workers, request_text, headers.as_ref())
    }

    fn reset(&self) {
        self.inner.reset();
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        "ConsistentHashPolicy()".to_string()
    }
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn add_to_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Worker info
    m.add_class::<PyWorkerInfo>()?;

    // Policies
    m.add_class::<PyRandomPolicy>()?;
    m.add_class::<PyRoundRobinPolicy>()?;
    m.add_class::<PyPowerOfTwoPolicy>()?;
    m.add_class::<PyConsistentHashPolicy>()?;

    Ok(())
}
