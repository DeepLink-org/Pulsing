//! Python bindings for the Pulsing Actor System

use pulsing_actor::actor::{ActorId, ActorPath, NodeId};
use pulsing_actor::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::net::SocketAddr;
use std::sync::Arc;

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

    fn from_message(msg: Message) -> Self {
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
            Ok(PyMessage::from_message(response))
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
                let py_msg = PyMessage::from_message(response);
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
        let handler = self.handler.clone();
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
        let handler = self.handler.clone();

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
        let handler = self.handler.clone();
        let event_loop = self.event_loop.clone();

        // Convert Message to PyMessage for Python
        let py_msg = PyMessage::from_message(msg);

        let response = python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<PyMessage> {
                    let receive_method = handler.getattr(py, "receive")?;
                    let result = receive_method.call1(py, (py_msg.clone(),))?;

                    let asyncio = py.import("asyncio")?;
                    let is_coro = asyncio
                        .call_method1("iscoroutine", (&result,))?
                        .extract::<bool>()?;

                    if is_coro {
                        let run_coroutine_threadsafe = asyncio.getattr("run_coroutine_threadsafe")?;
                        let future = run_coroutine_threadsafe.call1((&result, &event_loop))?;
                        let py_result = future.call_method0("result")?;

                        if py_result.is_none() {
                            Ok(PyMessage::empty())
                        } else {
                            let response: PyMessage = py_result.extract()?;
                            Ok(response)
                        }
                    } else {
                        if result.bind(py).is_none() {
                            Ok(PyMessage::empty())
                        } else {
                            let response: PyMessage = result.extract(py)?;
                            Ok(response)
                        }
                    }
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("Python executor error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Python handler error: {:?}", e))?;

        Ok(response.to_message())
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
    Ok(())
}
