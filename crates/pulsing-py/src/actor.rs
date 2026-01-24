//! Python bindings for the Pulsing Actor System

use futures::StreamExt;
use pulsing_actor::actor::{ActorId, ActorPath, NodeId};
use pulsing_actor::prelude::*;
use pulsing_actor::supervision::{BackoffStrategy, RestartPolicy, SupervisionSpec};
use pyo3::exceptions::{PyException, PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;

use crate::python_executor::python_executor;

/// Special message type identifier for pickle-encoded Python objects
const SEALED_PY_MSG_TYPE: &str = "__sealed_py_message__";

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
    fn new(id: u64) -> Self {
        Self {
            inner: NodeId::new(id),
        }
    }

    #[staticmethod]
    fn local() -> Self {
        Self {
            inner: NodeId::LOCAL,
        }
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.0
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("NodeId({})", self.inner.0)
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
    #[pyo3(signature = (local_id, node=None))]
    fn new(local_id: u64, node: Option<PyNodeId>) -> Self {
        let inner = match node {
            Some(n) => ActorId::new(n.inner, local_id),
            None => ActorId::local(local_id),
        };
        Self { inner }
    }

    #[staticmethod]
    fn local(local_id: u64) -> Self {
        Self {
            inner: ActorId::local(local_id),
        }
    }

    #[getter]
    fn local_id(&self) -> u64 {
        self.inner.local_id()
    }

    #[getter]
    fn node(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.node(),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ActorId(local_id={}, node={})",
            self.inner.local_id(),
            self.inner.node()
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

    /// Parse ActorId from string format "node_id:local_id"
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid ActorId format: '{}'. Expected 'node_id:local_id'",
                s
            )));
        }
        let node_id: u64 = parts[0].parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid node_id in ActorId: '{}'",
                parts[0]
            ))
        })?;
        let local_id: u64 = parts[1].parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid local_id in ActorId: '{}'",
                parts[1]
            ))
        })?;
        Ok(Self {
            inner: ActorId::new(NodeId::new(node_id), local_id),
        })
    }
}

/// Python wrapper for Message (unified, supports both single and stream)
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    msg_type: String,
    /// Payload for single messages (None for stream messages)
    payload: Option<Vec<u8>>,
    /// Stream reader for stream messages (None for single messages)
    stream_reader: Option<Arc<TokioMutex<Option<pulsing_actor::actor::MessageStream>>>>,
}

#[pymethods]
impl PyMessage {
    /// Create a single (non-streaming) message
    #[new]
    #[pyo3(signature = (msg_type, payload=None))]
    fn new(msg_type: String, payload: Option<Vec<u8>>) -> Self {
        Self {
            msg_type,
            payload: payload.or(Some(Vec::new())),
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
            format!("Message(msg_type='{}', stream=True)", self.msg_type)
        } else {
            format!(
                "Message(msg_type='{}', payload_len={})",
                self.msg_type,
                self.payload.as_ref().map(|p| p.len()).unwrap_or(0)
            )
        }
    }
}

impl PyMessage {
    /// Convert to Rust Message
    fn to_message(&self) -> Message {
        if self.stream_reader.is_some() {
            Message::single(&self.msg_type, Vec::new())
        } else {
            Message::single(&self.msg_type, self.payload.clone().unwrap_or_default())
        }
    }

    /// Create from Rust Message (supports both single and stream)
    fn from_rust_message(msg: Message) -> Self {
        match msg {
            Message::Single { msg_type, data } => Self {
                msg_type,
                payload: Some(data),
                stream_reader: None,
            },
            Message::Stream {
                default_msg_type,
                stream,
            } => Self {
                msg_type: default_msg_type,
                payload: None,
                stream_reader: Some(Arc::new(TokioMutex::new(Some(stream)))),
            },
        }
    }
}

// ============================================================================
// SealedPyMessage - Pickle-encoded Python objects for Python-to-Python communication
// ============================================================================

/// Pickle-encoded Python object wrapper for transparent Python object passing.
///
/// This allows Python actors to send and receive arbitrary Python objects
/// without the need for JSON serialization. The object is serialized using
/// Python's pickle module.
#[pyclass(name = "SealedPyMessage")]
#[derive(Clone)]
pub struct PySealedMessage {
    /// Pickle-encoded Python object bytes
    data: Vec<u8>,
}

#[pymethods]
impl PySealedMessage {
    /// Create a SealedPyMessage by pickling any Python object
    #[staticmethod]
    fn seal(py: Python<'_>, obj: PyObject) -> PyResult<Self> {
        let pickle = py.import("pickle")?;
        let dumped = pickle.call_method1("dumps", (&obj,))?;
        let bytes = dumped.downcast::<PyBytes>()?;
        Ok(Self {
            data: bytes.as_bytes().to_vec(),
        })
    }

    /// Unseal (unpickle) the message back to a Python object
    fn unseal(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pickle = py.import("pickle")?;
        let bytes = PyBytes::new(py, &self.data);
        let obj = pickle.call_method1("loads", (bytes,))?;
        Ok(obj.into())
    }

    /// Get raw pickle bytes
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    fn __repr__(&self) -> String {
        format!("SealedPyMessage(data_len={})", self.data.len())
    }
}

/// Helper function to pickle a Python object in Rust
fn pickle_object(py: Python<'_>, obj: &PyObject) -> PyResult<Vec<u8>> {
    let pickle = py.import("pickle")?;
    let dumped = pickle.call_method1("dumps", (obj,))?;
    let bytes = dumped.downcast::<PyBytes>()?;
    Ok(bytes.as_bytes().to_vec())
}

/// Helper function to unpickle bytes back to a Python object
fn unpickle_object(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let pickle = py.import("pickle")?;
    let bytes = PyBytes::new(py, data);
    let obj = pickle.call_method1("loads", (bytes,))?;
    Ok(obj.into())
}

// ============================================================================
// PyStreamReader - Async iterator for reading incoming streams
// ============================================================================

/// Async stream reader for consuming streaming messages from Rust.
///
/// Now reads `Message` items from the stream, providing access to msg_type per chunk.
#[pyclass(name = "StreamReader")]
pub struct PyStreamReader {
    stream: Arc<TokioMutex<Option<pulsing_actor::actor::MessageStream>>>,
}

#[pymethods]
impl PyStreamReader {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterate and return Python objects (auto unpickle if sealed)
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            if let Some(ref mut s) = *guard {
                match s.next().await {
                    Some(Ok(msg)) => Python::with_gil(|py| {
                        // Auto unpickle if it's a sealed Python message
                        match &msg {
                            Message::Single { msg_type, data }
                                if msg_type == SEALED_PY_MSG_TYPE =>
                            {
                                unpickle_object(py, data)
                            }
                            _ => {
                                // Return as PyMessage for JSON/other types
                                Ok(PyMessage::from_rust_message(msg)
                                    .into_pyobject(py)?
                                    .into_any()
                                    .unbind())
                            }
                        }
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

    /// Cancel the stream
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
/// Now sends `Message::Single` items, allowing each chunk to have its own msg_type.
#[pyclass(name = "StreamWriter")]
pub struct PyStreamWriter {
    #[allow(clippy::type_complexity)]
    sender: Arc<TokioMutex<Option<mpsc::Sender<anyhow::Result<Message>>>>>,
}

#[pymethods]
impl PyStreamWriter {
    /// Write any Python object to the stream (auto pickle)
    ///
    /// This is the recommended method for Python-to-Python streaming.
    /// Objects are automatically pickled and will be unpickled on the reader side.
    fn write<'py>(&self, py: Python<'py>, obj: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let pickled = pickle_object(py, &obj)?;
        let sender = self.sender.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = sender.lock().await;
            if let Some(ref tx) = *guard {
                let msg = Message::single(SEALED_PY_MSG_TYPE, pickled);
                tx.send(Ok(msg))
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
            *guard = None;
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
/// Now uses `Message` stream, allowing heterogeneous message types in the stream.
#[pyclass(name = "StreamMessage")]
pub struct PyStreamMessage {
    /// Default message type (used when chunk doesn't specify one)
    default_msg_type: String,
    #[allow(clippy::type_complexity)]
    receiver: Arc<StdMutex<Option<mpsc::Receiver<anyhow::Result<Message>>>>>,
}

#[pymethods]
impl PyStreamMessage {
    /// Create a new streaming message with a writer.
    ///
    /// The `msg_type` is the default message type for chunks that don't specify their own.
    #[staticmethod]
    #[pyo3(signature = (msg_type, buffer_size=32))]
    fn create(msg_type: String, buffer_size: usize) -> (PyStreamMessage, PyStreamWriter) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (
            PyStreamMessage {
                default_msg_type: msg_type,
                receiver: Arc::new(StdMutex::new(Some(rx))),
            },
            PyStreamWriter {
                sender: Arc::new(TokioMutex::new(Some(tx))),
            },
        )
    }

    #[getter]
    fn msg_type(&self) -> String {
        self.default_msg_type.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamMessage(default_msg_type='{}')",
            self.default_msg_type
        )
    }
}

/// Response type from Python actor - can be single, stream, or sealed (pickled)
enum PyActorResponse {
    Single(PyMessage),
    /// Stream of Messages with default msg_type
    StreamChannel(String, mpsc::Receiver<anyhow::Result<Message>>),
    /// Pickled Python object for Python-to-Python communication
    Sealed(Vec<u8>),
    /// Generator (async or sync) to be iterated
    Generator(PyObject, PyObject, bool), // (generator, event_loop, is_async)
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
            inner: *self.inner.id(),
        }
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    /// Send a message and wait for response (supports both single and stream responses)
    ///
    /// The message can be:
    /// - Any Python object: automatically pickled for Python-to-Python communication
    /// - Message: uses JSON encoding for Rust actor communication
    ///
    /// Returns:
    /// - For Python actors: the original Python object returned by receive()
    /// - For Rust actors: a Message object
    fn ask<'py>(&self, py: Python<'py>, msg: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();

        // Check if msg is already a PyMessage
        let msg_bound = msg.bind(py);
        let actor_msg = if msg_bound.is_instance_of::<PyMessage>() {
            let py_msg: PyMessage = msg_bound.extract()?;
            py_msg.to_message()
        } else {
            // Pickle any other Python object
            let pickled = pickle_object(py, &msg)?;
            Message::single(SEALED_PY_MSG_TYPE, pickled)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;

            // Check if response is a sealed message
            Python::with_gil(|py| {
                match response {
                    Message::Single {
                        ref msg_type,
                        ref data,
                    } if msg_type == SEALED_PY_MSG_TYPE => {
                        // Unpickle and return the original Python object
                        unpickle_object(py, data)
                    }
                    _ => {
                        // Return as PyMessage for non-sealed responses
                        Ok(PyMessage::from_rust_message(response)
                            .into_pyobject(py)?
                            .into_any()
                            .unbind())
                    }
                }
            })
        })
    }

    /// Send a message without waiting for response (fire-and-forget)
    ///
    /// The message can be:
    /// - Any Python object: automatically pickled for Python-to-Python communication
    /// - Message: uses JSON encoding for Rust actor communication
    fn tell<'py>(&self, py: Python<'py>, msg: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();

        // Check if msg is already a PyMessage
        let msg_bound = msg.bind(py);
        let actor_msg = if msg_bound.is_instance_of::<PyMessage>() {
            let py_msg: PyMessage = msg_bound.extract()?;
            py_msg.to_message()
        } else {
            // Pickle any other Python object
            let pickled = pickle_object(py, &msg)?;
            Message::single(SEALED_PY_MSG_TYPE, pickled)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            actor_ref.send_oneway(actor_msg).await.map_err(to_pyerr)?;
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

    /// Enable TLS with passphrase-derived certificates
    ///
    /// All nodes using the same passphrase will be able to communicate securely.
    /// The passphrase is used to derive a shared CA certificate, enabling
    /// automatic mutual TLS authentication.
    ///
    /// Example:
    ///     config = SystemConfig.with_addr("0.0.0.0:8000").with_passphrase("my-cluster-secret")
    #[cfg(feature = "tls")]
    fn with_passphrase(&self, passphrase: String) -> PyResult<Self> {
        let new_inner = self.inner.clone().with_tls(&passphrase).map_err(to_pyerr)?;
        Ok(Self { inner: new_inner })
    }

    /// Check if TLS is enabled
    fn is_tls_enabled(&self) -> bool {
        self.inner.is_tls_enabled()
    }

    fn __repr__(&self) -> String {
        let tls_status = if self.inner.is_tls_enabled() {
            ", tls=enabled"
        } else {
            ""
        };
        format!("SystemConfig(addr={}{})", self.inner.addr, tls_status)
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

#[async_trait::async_trait]
impl Actor for PythonActorWrapper {
    fn metadata(&self) -> std::collections::HashMap<String, String> {
        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();

            // First, try to extract built-in Python class information
            if let Ok(class) = self.handler.getattr(py, "__class__") {
                // Get class name
                if let Ok(name) = class.getattr(py, "__name__") {
                    if let Ok(name_str) = name.extract::<String>(py) {
                        result.insert("python_class".to_string(), name_str);
                    }
                }

                // Get module name
                if let Ok(module) = class.getattr(py, "__module__") {
                    if let Ok(module_str) = module.extract::<String>(py) {
                        result.insert("python_module".to_string(), module_str);
                    }
                }

                // Get source file path
                if let Ok(module_name) = class.getattr(py, "__module__") {
                    if let Ok(module_str) = module_name.extract::<String>(py) {
                        // Try to get the module and its file path
                        if let Ok(sys) = py.import("sys") {
                            if let Ok(modules) = sys.getattr("modules") {
                                if let Ok(module_obj) = modules.get_item(module_str.as_str()) {
                                    if let Ok(file_attr) = module_obj.getattr("__file__") {
                                        if let Ok(file_path) = file_attr.extract::<String>() {
                                            result.insert("python_file".to_string(), file_path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Then, check if the actor has custom metadata attribute
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
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let actor_id = *ctx.id();
        let event_loop = Python::with_gil(|py| self.event_loop.clone_ref(py));

        python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<()> {
                    if handler.getattr(py, "on_start").is_ok() {
                        let py_actor_id = PyActorId { inner: actor_id };
                        let result = handler.call_method1(py, "on_start", (py_actor_id,))?;

                        // Check if return value is a coroutine, if so wait for it to complete
                        let asyncio = py.import("asyncio")?;
                        let is_coro = asyncio
                            .call_method1("iscoroutine", (&result,))?
                            .extract::<bool>()?;

                        if is_coro {
                            let run_coroutine_threadsafe =
                                asyncio.getattr("run_coroutine_threadsafe")?;
                            let future = run_coroutine_threadsafe.call1((&result, &event_loop))?;
                            future.call_method0("result")?;
                        }

                        Ok(())
                    } else {
                        Ok(())
                    }
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Python on_start error: {:?}", e))
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let event_loop = Python::with_gil(|py| self.event_loop.clone_ref(py));

        python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<()> {
                    if handler.getattr(py, "on_stop").is_ok() {
                        let result = handler.call_method0(py, "on_stop")?;

                        // Check if return value is a coroutine, if so wait for it to complete
                        let asyncio = py.import("asyncio")?;
                        let is_coro = asyncio
                            .call_method1("iscoroutine", (&result,))?
                            .extract::<bool>()?;

                        if is_coro {
                            let run_coroutine_threadsafe =
                                asyncio.getattr("run_coroutine_threadsafe")?;
                            let future = run_coroutine_threadsafe.call1((&result, &event_loop))?;
                            future.call_method0("result")?;
                        }

                        Ok(())
                    } else {
                        Ok(())
                    }
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Python on_stop error: {:?}", e))
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let (handler, event_loop) =
            Python::with_gil(|py| (self.handler.clone_ref(py), self.event_loop.clone_ref(py)));

        // Check if this is a sealed Python message
        let is_sealed_msg = msg.msg_type() == SEALED_PY_MSG_TYPE;
        let py_msg = PyMessage::from_rust_message(msg);

        let response = python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<PyActorResponse> {
                    let receive_method = handler.getattr(py, "receive")?;

                    // If sealed message, unpickle and pass the original Python object
                    let call_arg: PyObject = if is_sealed_msg {
                        let payload = py_msg.payload.as_ref().ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("Expected payload for sealed message")
                        })?;
                        unpickle_object(py, payload)?
                    } else {
                        py_msg.into_pyobject(py)?.into_any().unbind()
                    };

                    let result = receive_method.call1(py, (call_arg,))?;

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

                    let py_result_bound = py_result.bind(py);

                    if py_result_bound.is_none() {
                        return Ok(PyActorResponse::Single(PyMessage::empty()));
                    }

                    // Check for generator (sync or async) - fast path using type name
                    let type_name = py_result_bound
                        .get_type()
                        .qualname()
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let is_gen = type_name == "generator";
                    let is_async_gen = type_name == "async_generator";

                    if is_async_gen || is_gen {
                        return Ok(PyActorResponse::Generator(
                            py_result.clone_ref(py),
                            event_loop.clone_ref(py),
                            is_async_gen,
                        ));
                    }

                    // Handle StreamMessage
                    if py_result_bound.is_instance_of::<PyStreamMessage>() {
                        let stream_msg_cell = py_result_bound.downcast::<PyStreamMessage>()?;

                        let borrowed = stream_msg_cell.borrow();
                        let default_msg_type = borrowed.default_msg_type.clone();
                        let receiver_arc = borrowed.receiver.clone();
                        drop(borrowed);

                        let receiver = {
                            let mut guard = receiver_arc.lock().map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "Lock error: {}",
                                    e
                                ))
                            })?;
                            guard.take()
                        };

                        if let Some(rx) = receiver {
                            return Ok(PyActorResponse::StreamChannel(default_msg_type, rx));
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "StreamMessage receiver already consumed",
                            ));
                        }
                    }

                    // Handle PyMessage (for Rust actor communication)
                    if py_result_bound.is_instance_of::<PyMessage>() {
                        let msg: PyMessage = py_result_bound.extract()?;
                        if msg.is_stream() {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "PyMessage with stream cannot be returned from receive(), use StreamMessage instead"
                            ));
                        } else {
                            return Ok(PyActorResponse::Single(msg));
                        }
                    }

                    // For any other Python object, pickle it and return as SealedPyMessage
                    let pickled = pickle_object(py, &py_result)?;
                    Ok(PyActorResponse::Sealed(pickled))
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Python handler error: {:?}", e))?;

        match response {
            PyActorResponse::Single(msg) => Ok(msg.to_message()),
            PyActorResponse::StreamChannel(default_msg_type, rx) => {
                Ok(Message::from_channel(&default_msg_type, rx))
            }
            PyActorResponse::Sealed(data) => Ok(Message::single(SEALED_PY_MSG_TYPE, data)),
            PyActorResponse::Generator(generator, event_loop, is_async) => {
                // Create channel for streaming generator values
                let (tx, rx) = mpsc::channel(32);

                // Spawn background task to iterate generator
                tokio::spawn(async move {
                    let result = python_executor()
                        .execute(move || {
                            Python::with_gil(|py| -> PyResult<()> {
                                let gen = generator.bind(py);
                                let asyncio = py.import("asyncio")?;

                                if is_async {
                                    // Async generator: iterate using anext()
                                    let run_coroutine_threadsafe =
                                        asyncio.getattr("run_coroutine_threadsafe")?;
                                    loop {
                                        let anext_coro = gen.call_method0("__anext__")?;
                                        let future = run_coroutine_threadsafe
                                            .call1((&anext_coro, &event_loop))?;
                                        match future.call_method0("result") {
                                            Ok(item) => {
                                                let pickled = pickle_object(py, &item.unbind())?;
                                                let msg =
                                                    Message::single(SEALED_PY_MSG_TYPE, pickled);
                                                if tx.blocking_send(Ok(msg)).is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                // Check if StopAsyncIteration
                                                if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                                    break;
                                                }
                                                let _ = tx.blocking_send(Err(anyhow::anyhow!(
                                                    "Generator error: {}",
                                                    e
                                                )));
                                                break;
                                            }
                                        }
                                    }
                                } else {
                                    // Sync generator: iterate using next()
                                    loop {
                                        match gen.call_method0("__next__") {
                                            Ok(item) => {
                                                let pickled = pickle_object(py, &item.unbind())?;
                                                let msg =
                                                    Message::single(SEALED_PY_MSG_TYPE, pickled);
                                                if tx.blocking_send(Ok(msg)).is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                // Check if StopIteration
                                                if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                                    break;
                                                }
                                                let _ = tx.blocking_send(Err(anyhow::anyhow!(
                                                    "Generator error: {}",
                                                    e
                                                )));
                                                break;
                                            }
                                        }
                                    }
                                }
                                Ok(())
                            })
                        })
                        .await;

                    if let Err(e) = result {
                        tracing::error!("Generator iteration error: {:?}", e);
                    }
                });

                Ok(Message::from_channel(SEALED_PY_MSG_TYPE, rx))
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
        let config_inner = config.inner;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let system = ActorSystem::new(config_inner).await.map_err(to_pyerr)?;
            Ok(PyActorSystem {
                inner: system,
                event_loop,
            })
        })
    }

    #[getter]
    fn node_id(&self) -> PyNodeId {
        PyNodeId {
            inner: *self.inner.node_id(),
        }
    }

    #[getter]
    fn addr(&self) -> String {
        self.inner.addr().to_string()
    }

    #[pyo3(signature = (
        actor,
        *,
        name=None,
        public=false,
        restart_policy="never",
        max_restarts=3,
        min_backoff=0.1,
        max_backoff=30.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn spawn<'py>(
        &self,
        py: Python<'py>,
        actor: PyObject,
        name: Option<String>,
        public: bool,
        restart_policy: &str,
        max_restarts: u32,
        min_backoff: f64,
        max_backoff: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();
        let event_loop = self.event_loop.clone();

        // Parse supervision config
        let policy = match restart_policy.to_lowercase().as_str() {
            "always" => RestartPolicy::Always,
            "on-failure" | "on_failure" => RestartPolicy::OnFailure,
            _ => RestartPolicy::Never,
        };

        let supervision = if matches!(policy, RestartPolicy::Never) {
            SupervisionSpec::never()
        } else {
            SupervisionSpec {
                policy,
                max_restarts,
                backoff: BackoffStrategy::exponential(
                    std::time::Duration::from_secs_f64(min_backoff),
                    std::time::Duration::from_secs_f64(max_backoff),
                ),
                ..Default::default()
            }
        };

        // Extract Python class metadata
        let metadata = Python::with_gil(|py| {
            let mut meta = std::collections::HashMap::new();

            // Try to get original class info from _WrappedActor first
            // This handles the @remote decorator case where we wrap user classes
            let (module, qualname, file) = {
                // Check for __original_module__, __original_qualname__, __original_file__
                let orig_module = actor
                    .getattr(py, "__original_module__")
                    .ok()
                    .and_then(|m| m.extract::<String>(py).ok());
                let orig_qualname = actor
                    .getattr(py, "__original_qualname__")
                    .ok()
                    .and_then(|q| q.extract::<String>(py).ok());
                let orig_file = actor
                    .getattr(py, "__original_file__")
                    .ok()
                    .and_then(|f| f.extract::<String>(py).ok());

                if orig_module.is_some() || orig_qualname.is_some() {
                    (orig_module, orig_qualname, orig_file)
                } else {
                    // Fallback to regular class info
                    let class = actor
                        .getattr(py, "__class__")
                        .unwrap_or_else(|_| actor.clone_ref(py));

                    let module = class
                        .getattr(py, "__module__")
                        .ok()
                        .and_then(|m| m.extract::<String>(py).ok());
                    let qualname = class
                        .getattr(py, "__qualname__")
                        .ok()
                        .and_then(|q| q.extract::<String>(py).ok());

                    // Get __file__ from module
                    let file = module.as_ref().and_then(|module_str| {
                        py.import("sys")
                            .ok()
                            .and_then(|sys| sys.getattr("modules").ok())
                            .and_then(|modules| modules.get_item(module_str).ok())
                            .and_then(|mod_obj| mod_obj.getattr("__file__").ok())
                            .and_then(|f| f.extract::<String>().ok())
                    });

                    (module, qualname, file)
                }
            };

            if let Some(m) = module {
                meta.insert("module".to_string(), m);
            }
            if let Some(q) = qualname {
                meta.insert("class".to_string(), q);
            }
            if let Some(f) = file {
                meta.insert("file".to_string(), f);
            }

            meta
        });

        // Note: 'public' parameter is now ignored - all named actors are resolvable
        let _ = public;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let options = pulsing_actor::system::SpawnOptions::new()
                .supervision(supervision)
                .metadata(metadata);

            let actor_ref = match name {
                // Anonymous actor - no name provided (not resolvable)
                None => {
                    // Anonymous actors do not support supervision/restart
                    if !matches!(policy, RestartPolicy::Never) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Anonymous actors do not support supervision/restart. \
                             Provide a name to enable supervision, or set restart_policy='never'.",
                        ));
                    }
                    // actor is the instance
                    let actor_wrapper = PythonActorWrapper::new(actor, event_loop);
                    system
                        .spawn_anonymous_with_options(actor_wrapper, options)
                        .await
                        .map_err(to_pyerr)?
                }
                // Named actor (resolvable by name)
                Some(name) => {
                    // Ensure name follows namespace/name format
                    let name = if name.contains('/') {
                        name
                    } else {
                        format!("actors/{}", name)
                    };

                    // Parse the path - use new_system for system/* paths (internal use only)
                    let path = if name.starts_with("system/") {
                        ActorPath::new_system(&name).map_err(to_pyerr)?
                    } else {
                        ActorPath::new(&name).map_err(to_pyerr)?
                    };

                    if matches!(policy, RestartPolicy::Never) {
                        // actor is the instance
                        let actor_wrapper = PythonActorWrapper::new(actor, event_loop);
                        system
                            .spawn_named_with_options(path, actor_wrapper, options)
                            .await
                            .map_err(to_pyerr)?
                    } else {
                        // actor is a factory - named actor with supervision
                        let factory = move || {
                            Python::with_gil(|py| -> anyhow::Result<PythonActorWrapper> {
                                let event_loop = event_loop.clone_ref(py);
                                let instance = actor.call0(py).map_err(|e| {
                                    anyhow::anyhow!("Python factory error: {:?}", e)
                                })?;
                                Ok(PythonActorWrapper::new(instance, event_loop))
                            })
                        };
                        system
                            .spawn_named_factory(path, factory, options)
                            .await
                            .map_err(to_pyerr)?
                    }
                }
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

    /// Alias for actor_ref - get actor reference by ID
    fn refer<'py>(&self, py: Python<'py>, actor_id: PyActorId) -> PyResult<Bound<'py, PyAny>> {
        self.actor_ref(py, actor_id)
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

    /// Get all instances of a named actor across the cluster (with detailed info)
    fn get_named_instances<'py>(
        &self,
        py: Python<'py>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Ensure name follows namespace/name format
            let name = if name.contains('/') {
                name
            } else {
                format!("actors/{}", name)
            };
            // Use new_system for system/* paths (internal use)
            let path = if name.starts_with("system/") {
                ActorPath::new_system(&name).map_err(to_pyerr)?
            } else {
                ActorPath::new(&name).map_err(to_pyerr)?
            };
            let instances = system.get_named_instances_detailed(&path).await;
            let result: Vec<std::collections::HashMap<String, serde_json::Value>> = instances
                .into_iter()
                .map(|(member, instance_opt)| {
                    let mut map = std::collections::HashMap::new();
                    map.insert(
                        "node_id".to_string(),
                        serde_json::Value::String(member.node_id.to_string()),
                    );
                    map.insert(
                        "addr".to_string(),
                        serde_json::Value::String(member.addr.to_string()),
                    );
                    map.insert(
                        "status".to_string(),
                        serde_json::Value::String(format!("{:?}", member.status)),
                    );

                    // Add detailed instance info if available
                    if let Some(inst) = instance_opt {
                        map.insert(
                            "actor_id".to_string(),
                            serde_json::Value::String(inst.actor_id.to_string()),
                        );
                        // Add metadata fields
                        for (k, v) in inst.metadata {
                            map.insert(k, serde_json::Value::String(v));
                        }
                    }

                    map
                })
                .collect();

            Python::with_gil(|py| -> PyResult<PyObject> {
                use pythonize::pythonize;
                let pyobj = pythonize(py, &result)?;
                Ok(pyobj.into())
            })
        })
    }

    /// Get all named actors in the cluster
    fn all_named_actors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let all_named = system.all_named_actors().await;

            Python::with_gil(|py| -> PyResult<PyObject> {
                use pythonize::pythonize;
                let result: Vec<std::collections::HashMap<String, serde_json::Value>> = all_named
                    .into_iter()
                    .map(|info| {
                        let mut map = std::collections::HashMap::new();
                        map.insert(
                            "path".to_string(),
                            serde_json::Value::String(info.path.as_str().to_string()),
                        );
                        map.insert(
                            "instance_count".to_string(),
                            serde_json::Value::Number(serde_json::Number::from(
                                info.instance_count(),
                            )),
                        );
                        // Convert instance_nodes (HashSet<NodeId>) to list of node IDs as strings
                        let instances: Vec<serde_json::Value> = info
                            .instance_nodes
                            .iter()
                            .map(|id| serde_json::Value::String(id.to_string()))
                            .collect();
                        map.insert("instances".to_string(), serde_json::Value::Array(instances));

                        // Add detailed instance info if available
                        let detailed: Vec<serde_json::Value> = info
                            .instances
                            .iter()
                            .map(|(node_id, inst)| {
                                let mut inst_map = serde_json::Map::new();
                                inst_map.insert(
                                    "node_id".to_string(),
                                    serde_json::Value::String(node_id.to_string()),
                                );
                                inst_map.insert(
                                    "actor_id".to_string(),
                                    serde_json::Value::String(inst.actor_id.to_string()),
                                );
                                // Add metadata
                                for (k, v) in &inst.metadata {
                                    inst_map
                                        .insert(k.clone(), serde_json::Value::String(v.clone()));
                                }
                                serde_json::Value::Object(inst_map)
                            })
                            .collect();
                        if !detailed.is_empty() {
                            map.insert(
                                "detailed_instances".to_string(),
                                serde_json::Value::Array(detailed),
                            );
                        }

                        map
                    })
                    .collect();
                let pyobj = pythonize(py, &result)?;
                Ok(pyobj.into())
            })
        })
    }

    /// Resolve a named actor (selects one instance using load balancing)
    #[pyo3(signature = (name, node_id=None))]
    fn resolve_named<'py>(
        &self,
        py: Python<'py>,
        name: String,
        node_id: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Ensure name follows namespace/name format
            let name = if name.contains('/') {
                name
            } else {
                format!("actors/{}", name)
            };
            // Use new_system for system/* paths (internal use)
            let path = if name.starts_with("system/") {
                ActorPath::new_system(&name).map_err(to_pyerr)?
            } else {
                ActorPath::new(&name).map_err(to_pyerr)?
            };
            let node = node_id.map(NodeId::new);
            let actor_ref = system
                .resolve_named(&path, node.as_ref())
                .await
                .map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    /// Alias for resolve_named - resolve actor by name
    #[pyo3(signature = (name, *, node_id=None))]
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        name: String,
        node_id: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.resolve_named(py, name, node_id)
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

    /// Get the SystemActor reference
    fn system<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_ref = system.system().await.map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    /// Get remote SystemActor reference (for remote nodes)
    fn remote_system<'py>(&self, py: Python<'py>, node_id: u64) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Use system/core - the correct system actor path
            let path = ActorPath::new_system("system/core").map_err(to_pyerr)?;
            let actor_ref = system
                .resolve_named(&path, Some(&NodeId::new(node_id)))
                .await
                .map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
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
    // Sealed message support (for Python-to-Python communication)
    m.add_class::<PySealedMessage>()?;
    Ok(())
}
