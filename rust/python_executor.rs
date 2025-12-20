//! Python Executor - A dedicated thread pool for Python code execution
//!
//! This module provides a dedicated thread pool for executing Python code,
//! avoiding GIL contention issues with Tokio's async runtime.
//!
//! ## Why a dedicated thread pool?
//!
//! Python's GIL (Global Interpreter Lock) means that only one thread can execute
//! Python bytecode at a time. Using `tokio::task::spawn_blocking` for Python code
//! has several drawbacks:
//!
//! 1. **Shared resources**: Tokio's blocking pool is shared with other blocking tasks
//! 2. **Thread bloat**: The pool may spawn many threads, all waiting for the GIL
//! 3. **GIL contention**: More threads competing for the GIL increases overhead
//!
//! A dedicated Python thread pool with a small, fixed number of threads (default: 4)
//! provides better resource isolation and GIL-aware scheduling.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use tokio::sync::oneshot;

/// Global Python executor instance
static PYTHON_EXECUTOR: OnceLock<PythonExecutor> = OnceLock::new();

/// Default number of threads for Python execution
const DEFAULT_PYTHON_THREADS: usize = 4;

/// A task that can be sent to the Python executor
type PythonTask = Box<dyn FnOnce() + Send + 'static>;

/// A dedicated thread pool for executing Python code
///
/// This executor maintains a fixed number of threads specifically for Python
/// code execution, keeping them separate from Tokio's async runtime threads.
pub struct PythonExecutor {
    // Wrapped in Mutex because std::sync::mpsc::Sender is not Sync,
    // but we need PythonExecutor to be Sync for OnceLock<PythonExecutor>
    sender: Mutex<Sender<PythonTask>>,
    _threads: Vec<JoinHandle<()>>,
}

impl PythonExecutor {
    /// Create a new Python executor with the specified number of threads
    ///
    /// # Arguments
    /// * `num_threads` - Number of worker threads (default: 4)
    ///
    /// # Note
    /// Due to Python's GIL, more threads don't necessarily mean better performance
    /// for pure Python code. However, if Python code releases the GIL (e.g., during
    /// I/O or when calling C extensions like NumPy), multiple threads can be beneficial.
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<PythonTask>();
        let receiver = Arc::new(Mutex::new(receiver));

        let threads: Vec<_> = (0..num_threads)
            .map(|i| {
                let rx = receiver.clone();
                thread::Builder::new()
                    .name(format!("python-executor-{}", i))
                    .spawn(move || {
                        tracing::debug!("Python executor thread {} started", i);
                        loop {
                            let task = {
                                let guard = rx.lock().unwrap();
                                guard.recv()
                            };

                            match task {
                                Ok(task) => task(),
                                Err(_) => {
                                    // Channel closed, exit the thread
                                    tracing::debug!("Python executor thread {} shutting down", i);
                                    break;
                                }
                            }
                        }
                    })
                    .expect("Failed to spawn Python executor thread")
            })
            .collect();

        tracing::info!(
            "Python executor initialized with {} threads",
            num_threads
        );

        Self {
            sender: Mutex::new(sender),
            _threads: threads,
        }
    }

    /// Execute a function on the Python thread pool and await its result
    ///
    /// This method sends the task to a dedicated Python thread, avoiding
    /// blocking Tokio's async runtime while waiting for the GIL.
    ///
    /// # Arguments
    /// * `f` - A closure that will be executed on a Python thread
    ///
    /// # Returns
    /// The result of the closure, wrapped in a Result
    ///
    /// # Example
    /// ```ignore
    /// let result = executor.execute(|| {
    ///     Python::with_gil(|py| {
    ///         // Python operations here
    ///     })
    /// }).await?;
    /// ```
    pub async fn execute<F, R>(&self, f: F) -> Result<R, ExecutorError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let task: PythonTask = Box::new(move || {
            let result = f();
            // Ignore send error - receiver may have been dropped (e.g., timeout)
            let _ = tx.send(result);
        });

        self.sender
            .lock()
            .map_err(|_| ExecutorError::ChannelClosed)?
            .send(task)
            .map_err(|_| ExecutorError::ChannelClosed)?;

        rx.await.map_err(|_| ExecutorError::TaskCancelled)
    }

    /// Execute a function that may fail, propagating both executor and task errors
    ///
    /// This is a convenience method for tasks that return a Result.
    pub async fn execute_fallible<F, R, E>(&self, f: F) -> Result<R, ExecutorError>
    where
        F: FnOnce() -> Result<R, E> + Send + 'static,
        R: Send + 'static,
        E: std::fmt::Display + Send + 'static,
    {
        let result = self.execute(f).await?;
        result.map_err(|e| ExecutorError::TaskFailed(e.to_string()))
    }
}

/// Errors that can occur during Python execution
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Python executor channel closed")]
    ChannelClosed,

    #[error("Python task was cancelled")]
    TaskCancelled,

    #[error("Python task failed: {0}")]
    TaskFailed(String),
}

/// Get or initialize the global Python executor
///
/// The executor is lazily initialized with `DEFAULT_PYTHON_THREADS` threads
/// on first access.
pub fn python_executor() -> &'static PythonExecutor {
    PYTHON_EXECUTOR.get_or_init(|| PythonExecutor::new(DEFAULT_PYTHON_THREADS))
}

/// Initialize the global Python executor with a custom number of threads
///
/// This must be called before any calls to `python_executor()`.
/// Returns `Err` if the executor has already been initialized.
///
/// # Arguments
/// * `num_threads` - Number of worker threads
pub fn init_python_executor(num_threads: usize) -> Result<(), &'static str> {
    PYTHON_EXECUTOR
        .set(PythonExecutor::new(num_threads))
        .map_err(|_| "Python executor already initialized")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_basic() {
        let executor = PythonExecutor::new(2);

        let result = executor.execute(|| 42).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_executor_concurrent() {
        let executor = Arc::new(PythonExecutor::new(4));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let exec = executor.clone();
                tokio::spawn(async move { exec.execute(move || i * 2).await.unwrap() })
            })
            .collect();

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let expected: Vec<_> = (0..10).map(|i| i * 2).collect();
        assert_eq!(results, expected);
    }
}

