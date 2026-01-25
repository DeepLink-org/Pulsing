//! Dedicated thread pool for Python code execution.
//!
//! Avoids GIL contention with Tokio's async runtime by isolating Python
//! execution to a fixed-size thread pool (default: 4 threads).

use crossbeam_channel::{unbounded, Sender};
use std::sync::OnceLock;
use std::thread::{self, JoinHandle};
use tokio::sync::oneshot;

static PYTHON_EXECUTOR: OnceLock<PythonExecutor> = OnceLock::new();

const DEFAULT_PYTHON_THREADS: usize = 4;

type PythonTask = Box<dyn FnOnce() + Send + 'static>;

/// Dedicated thread pool for Python code execution.
pub struct PythonExecutor {
    sender: Sender<PythonTask>,
    _threads: Vec<JoinHandle<()>>,
}

impl PythonExecutor {
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = unbounded::<PythonTask>();

        let threads: Vec<_> = (0..num_threads)
            .map(|i| {
                let rx = receiver.clone();
                thread::Builder::new()
                    .name(format!("python-executor-{}", i))
                    .spawn(move || {
                        tracing::debug!("Python executor thread {} started", i);
                        loop {
                            match rx.recv() {
                                Ok(task) => task(),
                                Err(_) => {
                                    tracing::debug!("Python executor thread {} shutting down", i);
                                    break;
                                }
                            }
                        }
                    })
                    .expect("Failed to spawn Python executor thread")
            })
            .collect();

        tracing::info!("Python executor initialized with {} threads", num_threads);

        Self {
            sender,
            _threads: threads,
        }
    }

    pub async fn execute<F, R>(&self, f: F) -> Result<R, ExecutorError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let task: PythonTask = Box::new(move || {
            let result = f();
            let _ = tx.send(result);
        });

        self.sender
            .send(task)
            .map_err(|_| ExecutorError::ChannelClosed)?;

        rx.await.map_err(|_| ExecutorError::TaskCancelled)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Python executor channel closed")]
    ChannelClosed,

    #[error("Python task was cancelled")]
    TaskCancelled,

    #[error("Python task failed: {0}")]
    TaskFailed(String),
}

pub fn python_executor() -> &'static PythonExecutor {
    PYTHON_EXECUTOR.get_or_init(|| PythonExecutor::new(DEFAULT_PYTHON_THREADS))
}

pub fn init_python_executor(num_threads: usize) -> Result<(), &'static str> {
    PYTHON_EXECUTOR
        .set(PythonExecutor::new(num_threads))
        .map_err(|_| "Python executor already initialized")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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

        let results: Vec<i32> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let expected: Vec<i32> = (0..10).map(|i| i * 2).collect();
        assert_eq!(results, expected);
    }
}
