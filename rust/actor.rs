//! Python bindings for the Pulsing Actor System
//!
//! This module exposes the actor system to Python, allowing users to:
//! - Create and manage ActorSystem
//! - Define actors in Python by subclassing PyActor
//! - Send messages between actors using ask/tell patterns
//! - Build distributed actor clusters

use pulsing_actor::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

/// Convert any error to PyErr
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
    /// Generate a new unique NodeId
    #[staticmethod]
    fn generate() -> Self {
        Self {
            inner: NodeId::generate(),
        }
    }

    /// Create from string
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
    /// Create a new ActorId with node and name
    #[new]
    #[pyo3(signature = (name, node=None))]
    fn new(name: String, node: Option<PyNodeId>) -> Self {
        let inner = match node {
            Some(n) => ActorId::new(n.inner, name),
            None => ActorId::local(name),
        };
        Self { inner }
    }

    /// Create a local actor id (node will be set when spawned)
    #[staticmethod]
    fn local(name: String) -> Self {
        Self {
            inner: ActorId::local(name),
        }
    }

    /// Get the actor name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the node id
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
        format!("ActorId(name='{}', node='{}')", self.inner.name, self.inner.node)
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

/// Python wrapper for RawMessage
#[pyclass(name = "RawMessage")]
#[derive(Clone)]
pub struct PyRawMessage {
    inner: RawMessage,
}

#[pymethods]
impl PyRawMessage {
    /// Create a new raw message
    #[new]
    fn new(msg_type: String, payload: Vec<u8>) -> Self {
        Self {
            inner: RawMessage {
                msg_type,
                payload,
            },
        }
    }

    /// Create from JSON-serializable Python object
    #[staticmethod]
    fn from_json(py: Python<'_>, msg_type: String, data: PyObject) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(&data.into_bound(py))?;
        let payload = serde_json::to_vec(&json_value).map_err(to_pyerr)?;
        Ok(Self {
            inner: RawMessage { msg_type, payload },
        })
    }

    /// Deserialize payload as JSON and return Python object
    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value: serde_json::Value =
            serde_json::from_slice(&self.inner.payload).map_err(to_pyerr)?;
        let pyobj = pythonize::pythonize(py, &value)?;
        Ok(pyobj.into())
    }

    /// Create an empty response message
    #[staticmethod]
    fn empty() -> Self {
        Self {
            inner: RawMessage::empty(),
        }
    }

    /// Get the message type
    #[getter]
    fn msg_type(&self) -> String {
        self.inner.msg_type.clone()
    }

    /// Get the raw payload bytes
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.payload)
    }

    fn __repr__(&self) -> String {
        format!(
            "RawMessage(msg_type='{}', payload_len={})",
            self.inner.msg_type,
            self.inner.payload.len()
        )
    }
}

/// Python wrapper for ActorRef
#[pyclass(name = "ActorRef")]
#[derive(Clone)]
pub struct PyActorRef {
    inner: ActorRef,
}

#[pymethods]
impl PyActorRef {
    /// Get the actor ID
    #[getter]
    fn actor_id(&self) -> PyActorId {
        PyActorId {
            inner: self.inner.id().clone(),
        }
    }

    /// Check if this is a local reference
    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    /// Ask pattern - send a message and wait for response (async)
    /// 
    /// Args:
    ///     msg: RawMessage to send
    /// 
    /// Returns:
    ///     RawMessage response from the actor
    fn ask<'py>(&self, py: Python<'py>, msg: PyRawMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        let raw_msg = msg.inner;
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send_raw(raw_msg).await.map_err(to_pyerr)?;
            Ok(PyRawMessage { inner: response })
        })
    }

    /// Ask pattern with JSON data - send and wait for JSON response (async)
    /// 
    /// Args:
    ///     msg_type: Message type string
    ///     data: JSON-serializable Python object
    /// 
    /// Returns:
    ///     Python object (deserialized from JSON response)
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
            let raw_msg = RawMessage { msg_type, payload };
            let response = actor_ref.send_raw(raw_msg).await.map_err(to_pyerr)?;
            
            Python::with_gil(|py| -> PyResult<PyObject> {
                let value: serde_json::Value =
                    serde_json::from_slice(&response.payload).map_err(to_pyerr)?;
                let pyobj = pythonize::pythonize(py, &value)?;
                Ok(pyobj.into())
            })
        })
    }

    /// Tell pattern - send a message without waiting for response (async)
    fn tell<'py>(&self, py: Python<'py>, msg: PyRawMessage) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();
        let raw_msg = msg.inner;
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            actor_ref.tell_raw(raw_msg).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Tell pattern with JSON data (async)
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
            let raw_msg = RawMessage { msg_type, payload };
            actor_ref.tell_raw(raw_msg).await.map_err(to_pyerr)?;
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
    /// Create a standalone configuration (no cluster)
    #[staticmethod]
    fn standalone() -> Self {
        Self {
            inner: SystemConfig::standalone(),
        }
    }

    /// Create configuration with specific address
    #[staticmethod]
    fn with_addr(addr: String) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr.parse().map_err(to_pyerr)?;
        Ok(Self {
            inner: SystemConfig::with_addr(socket_addr),
        })
    }

    /// Add seed nodes for cluster joining
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

/// Internal Python actor that wraps a Python callable/object
struct PythonActorWrapper {
    id: ActorId,
    handler: PyObject,
    event_loop: PyObject,
}

impl PythonActorWrapper {
    fn new(id: ActorId, handler: PyObject, event_loop: PyObject) -> Self {
        Self {
            id,
            handler,
            event_loop,
        }
    }
}

#[async_trait]
impl Actor for PythonActorWrapper {
    fn id(&self) -> &ActorId {
        &self.id
    }

    fn metadata(&self) -> std::collections::HashMap<String, String> {
        Python::with_gil(|py| {
            let mut result = std::collections::HashMap::new();
            
            // Check if handler has 'metadata' method or property
            if let Ok(metadata_attr) = self.handler.getattr(py, "metadata") {
                let bound = metadata_attr.bind(py);
                
                // If it's callable, call it; otherwise use it directly
                let value = if bound.is_callable() {
                    bound.call0().ok()
                } else {
                    Some(bound.clone())
                };

                if let Some(v) = value {
                    // Try to extract as dict
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

    async fn on_start(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Python::with_gil(|py| {
            if self.handler.getattr(py, "on_start").is_ok() {
                let actor_id = PyActorId {
                    inner: self.id.clone(),
                };
                // Call on_start if it exists
                if let Err(e) = self.handler.call_method1(py, "on_start", (actor_id,)) {
                    tracing::warn!("Python actor on_start error: {:?}", e);
                }
            }
            Ok(())
        })
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        Python::with_gil(|py| {
            if self.handler.getattr(py, "on_stop").is_ok() {
                if let Err(e) = self.handler.call_method0(py, "on_stop") {
                    tracing::warn!("Python actor on_stop error: {:?}", e);
                }
            }
            Ok(())
        })
    }

    async fn receive(
        &mut self,
        msg: RawMessage,
        _ctx: &mut ActorContext,
    ) -> anyhow::Result<RawMessage> {
        let handler = self.handler.clone();
        let event_loop = self.event_loop.clone();
        let py_msg = PyRawMessage { inner: msg };

        // Run the Python handler
        let result = Python::with_gil(|py| -> PyResult<RawMessage> {
            // Check if the handler has a 'receive' method
            let receive_method = handler.getattr(py, "receive")?;
            
            // Call the receive method
            let result = receive_method.call1(py, (py_msg.clone(),))?;
            
            // Check if the result is a coroutine (async function)
            let asyncio = py.import("asyncio")?;
            let is_coro = asyncio
                .call_method1("iscoroutine", (&result,))?
                .extract::<bool>()?;
            
            if is_coro {
                // Run the coroutine in the event loop
                let run_coroutine_threadsafe = asyncio.getattr("run_coroutine_threadsafe")?;
                let future = run_coroutine_threadsafe.call1((&result, &event_loop))?;
                let py_result = future.call_method0("result")?;
                
                // Extract the PyRawMessage from result
                if py_result.is_none() {
                    Ok(RawMessage::empty())
                } else {
                    let response: PyRawMessage = py_result.extract()?;
                    Ok(response.inner)
                }
            } else {
                // Synchronous result
                if result.bind(py).is_none() {
                    Ok(RawMessage::empty())
                } else {
                    let response: PyRawMessage = result.extract(py)?;
                    Ok(response.inner)
                }
            }
        });

        result.map_err(|e| anyhow::anyhow!("Python handler error: {:?}", e))
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
    /// Create a new actor system (async)
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

    /// Get the local node ID
    #[getter]
    fn node_id(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.node_id().clone(),
        }
    }

    /// Get the system address
    #[getter]
    fn addr(&self) -> String {
        self.inner.addr().to_string()
    }

    /// Spawn a new Python actor
    /// 
    /// Args:
    ///     name: Actor name (must be unique within this node)
    ///     handler: Python object with a `receive(msg: RawMessage) -> RawMessage` method
    ///     public: Whether to broadcast this actor's existence to the cluster (default: False)
    /// 
    /// Returns:
    ///     ActorRef to the spawned actor
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
            let actor_id = ActorId::local(name);
            let actor = PythonActorWrapper::new(actor_id, handler, event_loop);
            
            let actor_ref = if public {
                system.spawn_named(actor).await.map_err(to_pyerr)?
            } else {
                system.spawn(actor).await.map_err(to_pyerr)?
            };
            
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    /// Get a reference to an actor (local or remote) (async)
    fn actor_ref<'py>(&self, py: Python<'py>, actor_id: PyActorId) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();
        let id = actor_id.inner;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_ref = system.actor_ref(&id).await.map_err(to_pyerr)?;
            Ok(PyActorRef { inner: actor_ref })
        })
    }

    /// Get cluster members (async)
    fn members<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let members = system.members().await;
            let result: Vec<HashMap<String, String>> = members
                .into_iter()
                .map(|m| {
                    let mut map = HashMap::new();
                    map.insert("node_id".to_string(), m.node_id.to_string());
                    map.insert("addr".to_string(), m.addr.to_string());
                    map.insert("status".to_string(), format!("{:?}", m.status));
                    map
                })
                .collect();
            Ok(result)
        })
    }

    /// Get local actor names
    fn local_actor_names(&self) -> Vec<String> {
        self.inner.local_actor_names()
    }

    /// Stop an actor (async)
    fn stop<'py>(&self, py: Python<'py>, actor_name: String) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            system.stop(&actor_name).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Shutdown the entire actor system (async)
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

/// Add actor module to the parent module
pub fn add_to_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyActorId>()?;
    m.add_class::<PyRawMessage>()?;
    m.add_class::<PyActorRef>()?;
    m.add_class::<PySystemConfig>()?;
    m.add_class::<PyActorSystem>()?;
    Ok(())
}

