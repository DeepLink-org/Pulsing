//! Python bindings for the Pulsing Actor System

use futures::StreamExt;
use pulsing_actor::actor::{ActorId, ActorPath, NodeId, PayloadStream};
use pulsing_actor::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;
use std::sync::Mutex as StdMutex;

use crate::python_executor::python_executor;

fn to_pyerr<E: std::fmt::Display>(err: E) -> PyErr {
    PyException::new_err(format!("{}", err))
}

/// Python wrapper for NodeId
#[pyclass(name = "NodeId")]
#[derive(Clone)]
pub struct PyNodeId {
    inner: NodeId,
}

#[pymethods]
impl PyNodeId {
    #[staticmethod]
    fn generate() -> Self {
        Self {
            inner: NodeId::generate(),
        }
    }

    #[new]
    fn new(id: String) -> Self {
        Self {
            inner: NodeId::new(id),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("NodeId('{}')", self.inner)
    }
}

/// Python wrapper for ActorId
#[pyclass(name = "ActorId")]
#[derive(Clone)]
pub struct PyActorId {
    inner: ActorId,
}

#[pymethods]
impl PyActorId {
    #[new]
    #[pyo3(signature = (name, node=None))]
    fn new(name: String, node: Option<PyNodeId>) -> Self {
        let inner = match node {
            Some(n) => ActorId::new(n.inner, name),
            None => ActorId::local(name),
        };
        Self { inner }
    }

    #[staticmethod]
    fn local(name: String) -> Self {
        Self {
            inner: ActorId::local(name),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn node(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.node.clone(),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ActorId(name='{}', node='{}')",
            self.inner.name, self.inner.node
        )
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &PyActorId) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for Message
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    msg_type: String,
    payload: Vec<u8>,
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(msg_type: String, payload: Vec<u8>) -> Self {
        Self { msg_type, payload }
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, msg_type: String, data: PyObject) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        Ok(Self { msg_type, payload })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value: serde_json::Value = serde_json::from_slice(&self.payload).map_err(to_pyerr)?;
        let pyobj = pythonize::pythonize(py, &value)?;
        Ok(pyobj.into())
    }

    #[staticmethod]
    fn empty() -> Self {
        Self {
            msg_type: String::new(),
            payload: Vec::new(),
        }
    }

    #[getter]
    fn msg_type(&self) -> String {
        self.msg_type.clone()
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(msg_type='{}', payload_len={})",
            self.msg_type,
            self.payload.len()
        )
    }
}

impl PyMessage {
    fn to_message(&self) -> Message {
        Message::single(&self.msg_type, self.payload.clone())
    }

    fn from_single(msg: Message) -> Self {
        match msg {
            Message::Single { msg_type, data } => Self {
                msg_type,
                payload: data,
            },
            Message::Stream { msg_type, .. } => Self {
                msg_type,
                payload: Vec::new(),
            },
        }
    }
}

// ============================================================================
// PyStreamReader - Async iterator for reading incoming streams
// ============================================================================

/// Async stream reader for consuming streaming messages from Rust.
///
/// Implements Python async iterator protocol for use with `async for`.
///
/// Example:
///     ```python
///     async for chunk in stream_reader:
///         data = json.loads(chunk)
///         process(data)
///     ```
#[pyclass(name = "StreamReader")]
pub struct PyStreamReader {
    stream: Arc<TokioMutex<Option<PayloadStream>>>,
}

impl PyStreamReader {
    fn new(stream: PayloadStream) -> Self {
        Self {
            stream: Arc::new(TokioMutex::new(Some(stream))),
        }
    }
}

#[pymethods]
impl PyStreamReader {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            if let Some(ref mut s) = *guard {
                match s.next().await {
                    Some(Ok(data)) => Python::with_gil(|py| {
                        Ok(PyBytes::new(py, &data).into_any().unbind())
                    }),
                    Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
                    None => {
                        *guard = None; // Mark as consumed
                        Err(PyStopAsyncIteration::new_err(""))
                    }
                }
            } else {
                Err(PyStopAsyncIteration::new_err("Stream already consumed"))
            }
        })
    }

    /// Read next chunk and parse as JSON
    fn read_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            if let Some(ref mut s) = *guard {
                match s.next().await {
                    Some(Ok(data)) => Python::with_gil(|py| {
                        let value: serde_json::Value =
                            serde_json::from_slice(&data).map_err(to_pyerr)?;
                        let pyobj = pythonize::pythonize(py, &value)?;
                        Ok(pyobj.unbind())
                    }),
                    Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
                    None => {
                        *guard = None;
                        Err(PyStopAsyncIteration::new_err(""))
                    }
                }
            } else {
                Err(PyStopAsyncIteration::new_err("Stream already consumed"))
            }
        })
    }

    /// Cancel the stream (notify sender to stop)
    fn cancel<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            *guard = None;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        "StreamReader()".to_string()
    }
}

// ============================================================================
// PyStreamWriter - Write data to outgoing stream
// ============================================================================

/// Stream writer for producing streaming responses.
///
/// Example:
///     ```python
///     stream_msg, writer = StreamMessage.create("TokenStream")
///     
///     async def produce():
///         for token in generate_tokens():
///             await writer.write_json({"text": token})
///         writer.close()
///     
///     asyncio.create_task(produce())
///     return stream_msg
///     ```
#[pyclass(name = "StreamWriter")]
pub struct PyStreamWriter {
    sender: Arc<TokioMutex<Option<mpsc::Sender<anyhow::Result<Vec<u8>>>>>>,
}

#[pymethods]
impl PyStreamWriter {
    /// Write raw bytes to the stream
    fn write<'py>(&self, py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let sender = self.sender.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = sender.lock().await;
            if let Some(ref tx) = *guard {
                tx.send(Ok(data))
                    .await
                    .map_err(|_| PyRuntimeError::new_err("Stream closed"))?;
                Ok(())
            } else {
                Err(PyRuntimeError::new_err("Writer already closed"))
            }
        })
    }

    /// Write an object as JSON to the stream
    fn write_json<'py>(&self, py: Python<'py>, data: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        let sender = self.sender.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = sender.lock().await;
            if let Some(ref tx) = *guard {
                tx.send(Ok(payload))
                    .await
                    .map_err(|_| PyRuntimeError::new_err("Stream closed"))?;
                Ok(())
            } else {
                Err(PyRuntimeError::new_err("Writer already closed"))
            }
        })
    }

    /// Close the stream normally
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sender = self.sender.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = sender.lock().await;
            *guard = None; // Drop sender to signal end
            Ok(())
        })
    }

    /// Close the stream with an error
    fn error<'py>(&self, py: Python<'py>, msg: String) -> PyResult<Bound<'py, PyAny>> {
        let sender = self.sender.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = sender.lock().await;
            if let Some(tx) = guard.take() {
                let _ = tx.send(Err(anyhow::anyhow!(msg))).await;
            }
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        "StreamWriter()".to_string()
    }
}

// ============================================================================
// PyStreamMessage - Streaming response message
// ============================================================================

/// Streaming message for returning stream responses from Python actors.
///
/// Create using `StreamMessage.create()` which returns both the message
/// and a writer for producing data.
///
/// Example:
///     ```python
///     stream_msg, writer = StreamMessage.create("TokenStream")
///     
///     async def produce():
///         for i in range(10):
///             await writer.write_json({"token": i})
///         writer.close()
///     
///     asyncio.create_task(produce())
///     return stream_msg
///     ```
#[pyclass(name = "StreamMessage")]
pub struct PyStreamMessage {
    msg_type: String,
    receiver: Arc<StdMutex<Option<mpsc::Receiver<anyhow::Result<Vec<u8>>>>>>,
}

#[pymethods]
impl PyStreamMessage {
    /// Create a new streaming message with a writer.
    ///
    /// Returns (StreamMessage, StreamWriter) tuple.
    /// Use the writer to produce data, then return the message from receive().
    #[staticmethod]
    #[pyo3(signature = (msg_type, buffer_size=32))]
    fn create(msg_type: String, buffer_size: usize) -> (PyStreamMessage, PyStreamWriter) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (
            PyStreamMessage {
                msg_type,
                receiver: Arc::new(StdMutex::new(Some(rx))),
            },
            PyStreamWriter {
                sender: Arc::new(TokioMutex::new(Some(tx))),
            },
        )
    }

    #[getter]
    fn msg_type(&self) -> String {
        self.msg_type.clone()
    }

    fn __repr__(&self) -> String {
        format!("StreamMessage(msg_type='{}')", self.msg_type)
    }
}


// ============================================================================
// PyUnifiedMessage - Unified message type supporting both single and stream
// ============================================================================

/// Unified message type that can be either single or streaming.
///
/// When receiving a message in Python, use `is_stream` to check the type,
/// then either access payload directly or get a stream reader.
///
/// Example:
///     ```python
///     async def receive(self, msg: UnifiedMessage) -> UnifiedMessage:
///         if msg.is_stream:
///             reader = msg.stream_reader()
///             async for chunk in reader:
///                 process(chunk)
///             return UnifiedMessage.single("Done", b"ok")
///         else:
///             # Handle single message
///             return UnifiedMessage.single("Echo", msg.payload)
///     ```
#[pyclass(name = "UnifiedMessage")]
pub struct PyUnifiedMessage {
    msg_type: String,
    payload: Option<Vec<u8>>,
    stream_reader: Option<Arc<TokioMutex<Option<PayloadStream>>>>,
}

#[pymethods]
impl PyUnifiedMessage {
    /// Create a single (non-streaming) message
    #[staticmethod]
    fn single(msg_type: String, payload: Vec<u8>) -> Self {
        Self {
            msg_type,
            payload: Some(payload),
            stream_reader: None,
        }
    }

    /// Create a single message from JSON data
    #[staticmethod]
    fn from_json(py: Python<'_>, msg_type: String, data: PyObject) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        Ok(Self {
            msg_type,
            payload: Some(payload),
            stream_reader: None,
        })
    }

    /// Create an empty message
    #[staticmethod]
    fn empty() -> Self {
        Self {
            msg_type: String::new(),
            payload: Some(Vec::new()),
            stream_reader: None,
        }
    }

    #[getter]
    fn msg_type(&self) -> String {
        self.msg_type.clone()
    }

    /// Check if this is a streaming message
    #[getter]
    fn is_stream(&self) -> bool {
        self.stream_reader.is_some()
    }

    /// Get payload bytes (only for single messages)
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match &self.payload {
            Some(data) => Ok(PyBytes::new(py, data)),
            None => Err(PyValueError::new_err(
                "Cannot get payload from stream message, use stream_reader() instead",
            )),
        }
    }

    /// Parse payload as JSON (only for single messages)
    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.payload {
            Some(data) => {
                let value: serde_json::Value = serde_json::from_slice(data).map_err(to_pyerr)?;
                let pyobj = pythonize::pythonize(py, &value)?;
                Ok(pyobj.into())
            }
            None => Err(PyValueError::new_err(
                "Cannot parse stream message as JSON, use stream_reader() instead",
            )),
        }
    }

    /// Get stream reader (only for stream messages)
    fn stream_reader(&self) -> PyResult<PyStreamReader> {
        match &self.stream_reader {
            Some(stream) => Ok(PyStreamReader {
                stream: stream.clone(),
            }),
            None => Err(PyValueError::new_err(
                "This is not a stream message, access payload directly",
            )),
        }
    }

    fn __repr__(&self) -> String {
        if self.is_stream() {
            format!("UnifiedMessage(msg_type='{}', stream=True)", self.msg_type)
        } else {
            format!(
                "UnifiedMessage(msg_type='{}', payload_len={})",
                self.msg_type,
                self.payload.as_ref().map(|p| p.len()).unwrap_or(0)
            )
        }
    }
}

impl PyUnifiedMessage {
    fn from_rust_message(msg: Message) -> Self {
        match msg {
            Message::Single { msg_type, data } => Self {
                msg_type,
                payload: Some(data),
                stream_reader: None,
            },
            Message::Stream { msg_type, stream } => Self {
                msg_type,
                payload: None,
                stream_reader: Some(Arc::new(TokioMutex::new(Some(stream)))),
            },
        }
    }
}

/// Response type from Python actor - can be single or stream
enum PyActorResponse {
    Single(PyMessage),
    StreamChannel(String, mpsc::Receiver<anyhow::Result<Vec<u8>>>),
}

/// Python wrapper for ActorRef
#[pyclass(name = "ActorRef")]
#[derive(Clone)]
pub struct PyActorRef {
    inner: ActorRef,
}

#[pymethods]
impl PyActorRef {
    #[getter]
    fn actor_id(&self) -> PyActorId {
        PyActorId {
            inner: self.inner.id().clone(),
        }
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    fn ask<'py>(&self, py: Python<'py>, msg: PyMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        let actor_msg = msg.to_message();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;
            Ok(PyMessage::from_single(response))
        })
    }

    #[pyo3(signature = (msg_type, data))]
    fn ask_json<'py>(
        &self,
        py: Python<'py>,
        msg_type: String,
        data: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        let actor_ref = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_msg = Message::single(&msg_type, payload);
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;

            Python::with_gil(|py| -> PyResult<PyObject> {
                let py_msg = PyMessage::from_single(response);
                let value: serde_json::Value =
                    serde_json::from_slice(&py_msg.payload).map_err(to_pyerr)?;
                let pyobj = pythonize::pythonize(py, &value)?;
                Ok(pyobj.into())
            })
        })
    }

    fn tell<'py>(&self, py: Python<'py>, msg: PyMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        let actor_msg = msg.to_message();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            actor_ref.fire(actor_msg).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Send a message and get a unified response (can be single or stream)
    fn send<'py>(&self, py: Python<'py>, msg: &PyUnifiedMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        
        // Convert PyUnifiedMessage to Rust Message
        let actor_msg = if let Some(ref data) = msg.payload {
            Message::single(&msg.msg_type, data.clone())
        } else {
            return Err(PyValueError::new_err(
                "Cannot send stream message as request (stream input not yet supported)",
            ));
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;
            Ok(PyUnifiedMessage::from_rust_message(response))
        })
    }

    /// Send a message and expect a streaming response
    fn ask_stream<'py>(&self, py: Python<'py>, msg: &PyUnifiedMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        
        // Convert PyUnifiedMessage to Rust Message
        let actor_msg = if let Some(ref data) = msg.payload {
            Message::single(&msg.msg_type, data.clone())
        } else {
            return Err(PyValueError::new_err(
                "Cannot send stream message as request (stream input not yet supported)",
            ));
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;
            match response {
                Message::Stream { stream, .. } => Ok(PyStreamReader::new(stream)),
                Message::Single { .. } => Err(PyValueError::new_err(
                    "Expected stream response but got single message",
                )),
            }
        })
    }

    #[pyo3(signature = (msg_type, data))]
    fn tell_json<'py>(
        &self,
        py: Python<'py>,
        msg_type: String,
        data: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        let actor_ref = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_msg = Message::single(&msg_type, payload);
            actor_ref.fire(actor_msg).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ActorRef(id={}, local={})",
            self.inner.id(),
            self.is_local()
        )
    }
}

/// Python wrapper for SystemConfig
#[pyclass(name = "SystemConfig")]
#[derive(Clone)]
pub struct PySystemConfig {
    inner: SystemConfig,
}

#[pymethods]
impl PySystemConfig {
    #[staticmethod]
    fn standalone() -> Self {
        Self {
            inner: SystemConfig::standalone(),
        }
    }

    #[staticmethod]
    fn with_addr(addr: String) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr.parse().map_err(to_pyerr)?;
        Ok(Self {
            inner: SystemConfig::with_addr(socket_addr),
        })
    }

    fn with_seeds(&self, seeds: Vec<String>) -> PyResult<Self> {
        let seed_addrs: Result<Vec<SocketAddr>, _> = seeds.iter().map(|s| s.parse()).collect();
        let seed_addrs = seed_addrs.map_err(to_pyerr)?;
        Ok(Self {
            inner: self.inner.clone().with_seeds(seed_addrs),
        })
    }

    fn __repr__(&self) -> String {
        format!("SystemConfig(addr={})", self.inner.addr)
    }
}

/// Python actor wrapper - bridges Python handler to Rust Actor trait
struct PythonActorWrapper {
    handler: PyObject,
    event_loop: PyObject,
}

impl PythonActorWrapper {
    fn new(handler: PyObject, event_loop: PyObject) -> Self {
        Self {
            handler,
            event_loop,
        }
    }
}

#[async_trait]
impl Actor for PythonActorWrapper {
    fn metadata(&self) -> std::collections::HashMap<String, String> {
        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();
            if let Ok(metadata_attr) = self.handler.getattr(py, "metadata") {
                let bound = metadata_attr.bind(py);
                let value = if bound.is_callable() {
                    bound.call0().ok()
                } else {
                    Some(bound.clone())
                };
                if let Some(v) = value {
                    if let Ok(dict) = v.downcast::<pyo3::types::PyDict>() {
                        for (k, val) in dict.iter() {
                            if let (Ok(key), Ok(value_str)) = (k.extract::<String>(), val.str()) {
                                result.insert(key, value_str.to_string());
                            }
                        }
                    }
                }
            }
            result
        })
    }

    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        // Clone handler inside GIL to avoid panic
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let actor_id = ctx.id().clone();

        python_executor()
            .execute(move || {
                Python::with_gil(|py| {
                    if handler.getattr(py, "on_start").is_ok() {
                        let py_actor_id = PyActorId { inner: actor_id };
                        if let Err(e) = handler.call_method1(py, "on_start", (py_actor_id,)) {
                            tracing::warn!("Python actor on_start error: {:?}", e);
                        }
                    }
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        // Clone handler inside GIL to avoid panic
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));

        python_executor()
            .execute(move || {
                Python::with_gil(|py| {
                    if handler.getattr(py, "on_stop").is_ok() {
                        if let Err(e) = handler.call_method0(py, "on_stop") {
                            tracing::warn!("Python actor on_stop error: {:?}", e);
                        }
                    }
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Clone Python objects inside GIL to avoid panic
        let (handler, event_loop) = Python::with_gil(|py| {
            (self.handler.clone_ref(py), self.event_loop.clone_ref(py))
        });

        // Convert Message to PyUnifiedMessage for Python (supports both single and stream)
        let py_msg = PyUnifiedMessage::from_rust_message(msg);

        let response = python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<PyActorResponse> {
                    let receive_method = handler.getattr(py, "receive")?;
                    let result = receive_method.call1(py, (py_msg,))?;

                    let asyncio = py.import("asyncio")?;
                    let is_coro = asyncio
                        .call_method1("iscoroutine", (&result,))?
                        .extract::<bool>()?;

                    let py_result = if is_coro {
                        let run_coroutine_threadsafe = asyncio.getattr("run_coroutine_threadsafe")?;
                        let future = run_coroutine_threadsafe.call1((&result, &event_loop))?;
                        future.call_method0("result")?.unbind()
                    } else {
                        result
                    };

                    // Check return type and convert appropriately
                    let py_result_bound = py_result.bind(py);
                    
                    if py_result_bound.is_none() {
                        return Ok(PyActorResponse::Single(PyMessage::empty()));
                    }

                    // Check if it's a StreamMessage (for streaming responses)
                    // Use downcast to check type and get mutable reference
                    if py_result_bound.is_instance_of::<PyStreamMessage>() {
                        // Get the PyStreamMessage from the cell
                        let stream_msg_cell = py_result_bound.downcast::<PyStreamMessage>()?;
                        
                        // Take the receiver from the stream message
                        let borrowed = stream_msg_cell.borrow();
                        let msg_type = borrowed.msg_type.clone();
                        let receiver_arc = borrowed.receiver.clone();
                        drop(borrowed); // Release borrow before locking
                        
                        // Use StdMutex - can lock synchronously
                        let receiver = {
                            let mut guard = receiver_arc.lock().map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
                            })?;
                            guard.take()
                        };
                        
                        if let Some(rx) = receiver {
                            return Ok(PyActorResponse::StreamChannel(msg_type, rx));
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "StreamMessage receiver already consumed"
                            ));
                        }
                    }

                    // Check if it's a UnifiedMessage
                    if py_result_bound.is_instance_of::<PyUnifiedMessage>() {
                        let unified = py_result_bound.downcast::<PyUnifiedMessage>()?;
                        let msg_type = unified.borrow().msg_type.clone();
                        
                        if let Some(ref data) = unified.borrow().payload {
                            return Ok(PyActorResponse::Single(PyMessage {
                                msg_type,
                                payload: data.clone(),
                            }));
                        } else {
                            // Stream response from UnifiedMessage is not supported here
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "UnifiedMessage with stream cannot be returned from receive()"
                            ));
                        }
                    }

                    // Finally try PyMessage (legacy support)
                    if let Ok(msg) = py_result_bound.extract::<PyMessage>() {
                        return Ok(PyActorResponse::Single(msg));
                    }

                    // If none match, return empty
                    Ok(PyActorResponse::Single(PyMessage::empty()))
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Python handler error: {:?}", e))?;

        // Convert response to Rust Message
        match response {
            PyActorResponse::Single(msg) => Ok(msg.to_message()),
            PyActorResponse::StreamChannel(msg_type, rx) => {
                Ok(Message::from_channel(&msg_type, rx))
            }
        }
    }
}

/// Python wrapper for ActorSystem
#[pyclass(name = "ActorSystem")]
pub struct PyActorSystem {
    inner: Arc<ActorSystem>,
    event_loop: PyObject,
}

#[pymethods]
impl PyActorSystem {
    #[staticmethod]
    fn create<'py>(
        py: Python<'py>,
        config: PySystemConfig,
        event_loop: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let system = ActorSystem::new(config.inner).await.map_err(to_pyerr)?;
            Ok(PyActorSystem {
                inner: system,
                event_loop,
            })
        })
    }

    #[getter]
    fn node_id(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.node_id().clone(),
        }
    }

    #[getter]
    fn addr(&self) -> String {
        self.inner.addr().to_string()
    }

    #[pyo3(signature = (name, handler, public=false))]
    fn spawn<'py>(
        &self,
        py: Python<'py>,
        name: String,
        handler: PyObject,
        public: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();
        let event_loop = self.event_loop.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor = PythonActorWrapper::new(handler, event_loop);

            let actor_ref = if public {
                let path = ActorPath::new(&format!("actors/{}", name)).map_err(to_pyerr)?;
                system
                    .spawn_named(path, &name, actor)
                    .await
                    .map_err(to_pyerr)?
            } else {
                system.spawn(&name, actor).await.map_err(to_pyerr)?
            };

            Ok(PyActorRef { inner: actor_ref })
        })
    }

    fn actor_ref<'py>(&self, py: Python<'py>, actor_id: PyActorId) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();
        let id = actor_id.inner;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_ref = system.actor_ref(&id).await.map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    fn members<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let members = system.members().await;
            let result: Vec<std::collections::HashMap<String, String>> = members
                .into_iter()
                .map(|m| {
                    let mut map = std::collections::HashMap::new();
                    map.insert("node_id".to_string(), m.node_id.to_string());
                    map.insert("addr".to_string(), m.addr.to_string());
                    map.insert("status".to_string(), format!("{:?}", m.status));
                    map
                })
                .collect();
            Ok(result)
        })
    }

    fn local_actor_names(&self) -> Vec<String> {
        self.inner.local_actor_names()
    }

    /// Get all instances of a named actor across the cluster
    fn get_named_instances<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let path = ActorPath::new(&format!("actors/{}", name)).map_err(to_pyerr)?;
            let instances: Vec<pulsing_actor::cluster::MemberInfo> = system.get_named_instances(&path).await;
            let result: Vec<std::collections::HashMap<String, String>> = instances
                .into_iter()
                .map(|m| {
                    let mut map = std::collections::HashMap::new();
                    map.insert("node_id".to_string(), m.node_id.to_string());
                    map.insert("addr".to_string(), m.addr.to_string());
                    map.insert("status".to_string(), format!("{:?}", m.status));
                    map
                })
                .collect();
            Ok(result)
        })
    }

    /// Resolve a named actor (selects one instance using load balancing)
    fn resolve_named<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let path = ActorPath::new(&format!("actors/{}", name)).map_err(to_pyerr)?;
            let addr = pulsing_actor::actor::ActorAddress::named(path);
            let actor_ref = system.resolve(&addr).await.map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    fn stop<'py>(&self, py: Python<'py>, actor_name: String) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            system.stop(&actor_name).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            system.shutdown().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ActorSystem(node_id='{}', addr='{}')",
            self.inner.node_id(),
            self.inner.addr()
        )
    }
}

pub fn add_to_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyActorId>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyActorRef>()?;
    m.add_class::<PySystemConfig>()?;
    m.add_class::<PyActorSystem>()?;
    // Streaming support
    m.add_class::<PyStreamReader>()?;
    m.add_class::<PyStreamWriter>()?;
    m.add_class::<PyStreamMessage>()?;
    m.add_class::<PyUnifiedMessage>()?;
    Ok(())
}
