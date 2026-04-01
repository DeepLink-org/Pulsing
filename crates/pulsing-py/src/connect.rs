//! Python bindings for the out-cluster PulsingConnect.

use pulsing_actor::connect::{ConnectActorRef, PulsingConnect};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::actor::PyMessage;

fn to_pyerr(err: pulsing_actor::error::PulsingError) -> PyErr {
    crate::errors::pulsing_error_to_py_err(err)
}

/// Out-cluster connector — connects to a gateway without joining the cluster.
#[pyclass(name = "PulsingConnect")]
pub struct PyPulsingConnect {
    inner: Arc<PulsingConnect>,
}

#[pymethods]
impl PyPulsingConnect {
    /// Connect to a single gateway address.
    #[staticmethod]
    fn connect<'py>(py: Python<'py>, addr: String) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let socket_addr: SocketAddr = addr
                .parse()
                .map_err(|e: std::net::AddrParseError| PyRuntimeError::new_err(e.to_string()))?;
            let conn = PulsingConnect::connect(socket_addr)
                .await
                .map_err(to_pyerr)?;
            Ok(PyPulsingConnect { inner: conn })
        })
    }

    /// Connect to multiple gateway addresses (for failover).
    #[staticmethod]
    fn connect_multi<'py>(py: Python<'py>, addrs: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let socket_addrs: Vec<SocketAddr> = addrs
                .iter()
                .map(|a| {
                    a.parse::<SocketAddr>()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let conn = PulsingConnect::connect_multi(socket_addrs)
                .await
                .map_err(to_pyerr)?;
            Ok(PyPulsingConnect { inner: conn })
        })
    }

    /// Resolve a named actor, returning a ConnectActorRef proxy.
    fn resolve<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let conn = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_ref = conn.resolve(&path).await.map_err(to_pyerr)?;
            Ok(PyConnectActorRef {
                inner: actor_ref,
                #[allow(clippy::redundant_clone)]
                _conn: conn.clone(),
            })
        })
    }

    /// Create a ConnectActorRef for a known path without resolve check.
    fn actor_ref<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let conn = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let gateway = conn.active_gateway().await;
            let actor_ref = ConnectActorRef::new(gateway, path, conn.http_client().clone());
            Ok(PyConnectActorRef {
                inner: actor_ref,
                #[allow(clippy::redundant_clone)]
                _conn: conn.clone(),
            })
        })
    }

    /// Refresh the gateway list from the cluster.
    fn refresh_gateways<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let conn = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            conn.refresh_gateways().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Get the current active gateway address.
    fn active_gateway<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let conn = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(conn.active_gateway().await.to_string())
        })
    }

    /// Close the connection.
    fn close(&self) {
        self.inner.close();
    }

    fn __repr__(&self) -> String {
        "PulsingConnect(out-cluster)".to_string()
    }
}

/// Python wrapper for ConnectActorRef.
#[pyclass(name = "ConnectActorRef")]
#[derive(Clone)]
pub struct PyConnectActorRef {
    inner: ConnectActorRef,
    #[allow(dead_code)]
    _conn: Arc<PulsingConnect>,
}

#[pymethods]
impl PyConnectActorRef {
    /// Send a message and receive a response (supports pickle protocol).
    fn ask<'py>(&self, py: Python<'py>, msg: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();

        let msg_bound = msg.bind(py);
        let actor_msg = if msg_bound.is_instance_of::<PyMessage>() {
            let py_msg: PyMessage = msg_bound.extract()?;
            py_msg.to_message()
        } else {
            crate::actor::encode_python_payload(py, &msg)?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send_message(actor_msg).await.map_err(to_pyerr)?;
            crate::actor::decode_message_to_pyobject(response).await
        })
    }

    /// Fire-and-forget message.
    fn tell<'py>(&self, py: Python<'py>, msg: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let actor_ref = self.inner.clone();

        let msg_bound = msg.bind(py);
        let actor_msg = if msg_bound.is_instance_of::<PyMessage>() {
            let py_msg: PyMessage = msg_bound.extract()?;
            py_msg.to_message()
        } else {
            crate::actor::encode_python_payload(py, &msg)?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match actor_msg {
                pulsing_actor::actor::Message::Single { msg_type, data } => {
                    actor_ref.tell(&msg_type, data).await.map_err(to_pyerr)?;
                }
                _ => {
                    return Err(PyRuntimeError::new_err("Streaming not supported for tell"));
                }
            }
            Ok(())
        })
    }

    /// Get the actor path.
    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string()
    }

    /// Get the gateway address.
    #[getter]
    fn gateway(&self) -> String {
        self.inner.gateway().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConnectActorRef(path='{}', gateway='{}')",
            self.inner.path(),
            self.inner.gateway()
        )
    }
}

pub fn add_to_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyPulsingConnect>()?;
    m.add_class::<PyConnectActorRef>()?;
    Ok(())
}
