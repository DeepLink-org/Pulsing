//! Python bindings for the Pulsing Actor System

use futures::StreamExt;
use pulsing_actor::actor::{ActorId, ActorPath, NodeId};
use pulsing_actor::prelude::*;
use pulsing_actor::supervision::{BackoffStrategy, RestartPolicy, SupervisionSpec};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;

use crate::errors::pulsing_error_to_py_err;
use crate::python_error_converter::convert_python_exception_to_actor_error;
use crate::python_executor::python_executor;

/// Special message type identifier for pickle-encoded Python objects
const SEALED_PY_MSG_TYPE: &str = "__sealed_py_message__";
/// Special message type identifier for zerocopy descriptor payloads (small, single message)
const SEALED_ZEROCOPY_MSG_TYPE: &str = "__sealed_zerocopy_message__";
/// Stream frame: descriptor header (metadata only, no bulk data)
const ZC_DESCRIPTOR_MSG_TYPE: &str = "__zc_descriptor__";
/// Stream frame: raw data chunk
const ZC_CHUNK_MSG_TYPE: &str = "__zc_chunk__";

/// Zerocopy metadata header — the single wire format for both single-message and stream paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZeroCopyDescriptorHeader {
    version: u32,
    buffer_count: usize,
    buffer_lengths: Vec<usize>,
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    strides: Option<Vec<isize>>,
    transport: Option<String>,
    checksum: Option<String>,
}

fn zerocopy_chunk_bytes() -> usize {
    const DEFAULT: usize = 1024 * 1024;
    const MIN: usize = 4 * 1024;
    std::env::var("PULSING_ZEROCOPY_CHUNK_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.max(MIN))
        .unwrap_or(DEFAULT)
}

fn zerocopy_stream_threshold() -> usize {
    const DEFAULT: usize = 64 * 1024;
    std::env::var("PULSING_ZEROCOPY_STREAM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT)
}

/// Convert PulsingError to Python exception (used for actor system APIs that return Result<_, PulsingError>).
fn to_pyerr(err: pulsing_actor::error::PulsingError) -> PyErr {
    pulsing_error_to_py_err(err)
}

/// Convert non-anyhow errors (parse, validation) to Python ValueError.
fn to_py_value_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
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

    /// Create a new NodeId from a u128 value or string UUID
    #[new]
    #[pyo3(signature = (id=None))]
    fn new(id: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<Self> {
        match id {
            None => Ok(Self {
                inner: NodeId::generate(),
            }),
            Some(py_id) => {
                // Try to extract as string first (UUID format)
                if let Ok(s) = py_id.extract::<String>() {
                    if let Ok(uuid) = uuid::Uuid::parse_str(&s) {
                        return Ok(Self {
                            inner: NodeId::new(uuid.as_u128()),
                        });
                    }
                }
                // Try as integer
                if let Ok(n) = py_id.extract::<u128>() {
                    return Ok(Self {
                        inner: NodeId::new(n),
                    });
                }
                // Try as smaller integer
                if let Ok(n) = py_id.extract::<u64>() {
                    return Ok(Self {
                        inner: NodeId::new(n as u128),
                    });
                }
                Err(PyValueError::new_err(
                    "NodeId must be a UUID string or integer",
                ))
            }
        }
    }

    #[staticmethod]
    fn local() -> Self {
        Self {
            inner: NodeId::LOCAL,
        }
    }

    /// Get the raw u128 value
    #[getter]
    fn id(&self) -> u128 {
        self.inner.0
    }

    /// Get the UUID string representation
    fn uuid(&self) -> String {
        self.inner.to_string()
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
    /// Create a new ActorId from a u128 value, string UUID, or generate a new one
    #[new]
    #[pyo3(signature = (id=None))]
    fn new(id: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<Self> {
        match id {
            None => Ok(Self {
                inner: ActorId::generate(),
            }),
            Some(py_id) => {
                // Try to extract as string first (UUID format)
                if let Ok(s) = py_id.extract::<String>() {
                    if let Ok(uuid) = uuid::Uuid::parse_str(&s) {
                        return Ok(Self {
                            inner: ActorId::new(uuid.as_u128()),
                        });
                    }
                }
                // Try as integer
                if let Ok(n) = py_id.extract::<u128>() {
                    return Ok(Self {
                        inner: ActorId::new(n),
                    });
                }
                // Try as smaller integer
                if let Ok(n) = py_id.extract::<u64>() {
                    return Ok(Self {
                        inner: ActorId::new(n as u128),
                    });
                }
                Err(PyValueError::new_err(
                    "ActorId must be a UUID string or integer",
                ))
            }
        }
    }

    /// Generate a new random ActorId
    #[staticmethod]
    fn generate() -> Self {
        Self {
            inner: ActorId::generate(),
        }
    }

    /// Get the raw u128 value
    #[getter]
    fn id(&self) -> u128 {
        self.inner.0
    }

    /// Get the UUID string representation
    fn uuid(&self) -> String {
        self.inner.to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ActorId({})", self.inner.0)
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

    /// Parse ActorId from string (UUID format)
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        // Try to parse as UUID
        if let Ok(uuid) = uuid::Uuid::parse_str(s) {
            return Ok(Self {
                inner: ActorId::new(uuid.as_u128()),
            });
        }
        // Try to parse as simple integer
        if let Ok(n) = s.parse::<u128>() {
            return Ok(Self {
                inner: ActorId::new(n),
            });
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid ActorId format: '{}'. Expected UUID string or integer",
            s
        )))
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
        let payload = serde_json::to_vec(&json_value).map_err(to_py_value_err)?;
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
                let value: serde_json::Value =
                    serde_json::from_slice(data).map_err(to_py_value_err)?;
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
    pub(crate) fn to_message(&self) -> Message {
        if self.stream_reader.is_some() {
            Message::single(&self.msg_type, Vec::new())
        } else {
            Message::single(&self.msg_type, self.payload.clone().unwrap_or_default())
        }
    }

    /// Create from Rust Message (supports both single and stream)
    pub(crate) fn from_rust_message(msg: Message) -> Self {
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

/// Descriptor object for optional zerocopy payload transport.
#[pyclass(name = "ZeroCopyDescriptor")]
#[derive(Clone)]
pub struct PyZeroCopyDescriptor {
    #[pyo3(get)]
    version: u32,
    #[pyo3(get)]
    buffers: Vec<PyObject>,
    #[pyo3(get)]
    dtype: Option<String>,
    #[pyo3(get)]
    shape: Option<Vec<usize>>,
    #[pyo3(get)]
    strides: Option<Vec<isize>>,
    #[pyo3(get)]
    transport: Option<String>,
    #[pyo3(get)]
    checksum: Option<String>,
}

/// Validate that a Python object exposes a contiguous buffer.
/// If it's not directly extractable as &[u8] (e.g. memoryview), convert via bytes().
fn ensure_contiguous_buffer(py: Python<'_>, item: &Bound<'_, pyo3::PyAny>) -> PyResult<PyObject> {
    if item.extract::<&[u8]>().is_ok() {
        return Ok(item.clone().unbind());
    }
    // Try converting via bytes() for memoryview and other buffer-protocol objects
    let builtins = py.import("builtins")?;
    let bytes_obj = builtins.getattr("bytes")?.call1((item,)).map_err(|_| {
        PyValueError::new_err(
            "ZeroCopyDescriptor.buffers items must expose a contiguous Python buffer (bytes/bytearray/memoryview/tensor)",
        )
    })?;
    // Verify the result is extractable
    bytes_obj.extract::<&[u8]>().map_err(|_| {
        PyValueError::new_err(
            "ZeroCopyDescriptor.buffers items must expose a contiguous Python buffer (bytes/bytearray/memoryview/tensor)",
        )
    })?;
    Ok(bytes_obj.unbind())
}

impl PyZeroCopyDescriptor {
    /// Total byte size of all buffers.
    fn total_buffer_bytes(&self, py: Python<'_>) -> usize {
        self.buffers
            .iter()
            .filter_map(|buf_obj| buf_obj.bind(py).extract::<&[u8]>().ok().map(|s| s.len()))
            .sum()
    }

    /// Build a descriptor header (metadata only, no data).
    fn to_header(&self, py: Python<'_>) -> ZeroCopyDescriptorHeader {
        ZeroCopyDescriptorHeader {
            version: self.version,
            buffer_count: self.buffers.len(),
            buffer_lengths: self
                .buffers
                .iter()
                .filter_map(|b| b.bind(py).extract::<&[u8]>().ok().map(|s| s.len()))
                .collect(),
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            transport: self.transport.clone(),
            checksum: self.checksum.clone(),
        }
    }

    /// Serialize for single-message path: [4-byte header_len LE] ++ header_bytes ++ raw_data.
    fn serialize_single(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        let header = self.to_header(py);
        let header_bytes = bincode::serialize(&header).map_err(to_py_value_err)?;
        let header_len = header_bytes.len() as u32;
        let total_data: usize = header.buffer_lengths.iter().sum();
        let mut out = Vec::with_capacity(4 + header_bytes.len() + total_data);
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for buf_obj in &self.buffers {
            let bound = buf_obj.bind(py);
            let data = bound.extract::<&[u8]>()?;
            out.extend_from_slice(data);
        }
        Ok(out)
    }

    /// Reconstruct from header + raw buffer data (shared by single and stream paths).
    fn from_wire(
        py: Python<'_>,
        header: ZeroCopyDescriptorHeader,
        raw_buffers: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            version: header.version,
            buffers: raw_buffers
                .into_iter()
                .map(|b| PyBytes::new(py, &b).into_any().unbind())
                .collect(),
            dtype: header.dtype,
            shape: header.shape,
            strides: header.strides,
            transport: header.transport,
            checksum: header.checksum,
        }
    }
}

#[pymethods]
impl PyZeroCopyDescriptor {
    #[new]
    #[pyo3(signature = (
        buffers,
        *,
        dtype=None,
        shape=None,
        strides=None,
        transport=None,
        checksum=None,
        version=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        buffers: Vec<PyObject>,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        strides: Option<Vec<isize>>,
        transport: Option<String>,
        checksum: Option<String>,
        version: u32,
    ) -> PyResult<Self> {
        if buffers.is_empty() {
            return Err(PyValueError::new_err(
                "ZeroCopyDescriptor requires at least one buffer",
            ));
        }
        let normalized: Vec<PyObject> = buffers
            .into_iter()
            .map(|item| ensure_contiguous_buffer(py, item.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            version,
            buffers: normalized,
            dtype,
            shape,
            strides,
            transport,
            checksum,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ZeroCopyDescriptor(version={}, buffers={}, transport={:?})",
            self.version,
            self.buffers.len(),
            self.transport
        )
    }
}

/// Helper function to pickle a Python object in Rust
fn pickle_object(py: Python<'_>, obj: &PyObject) -> PyResult<Vec<u8>> {
    let pickle = py.import("pickle")?;
    let dumped = pickle.call_method1("dumps", (obj,))?;
    let bytes = dumped.downcast::<PyBytes>()?;
    Ok(bytes.as_bytes().to_vec())
}

/// Try to extract a `PyZeroCopyDescriptor` from a Python object via `__zerocopy__(ctx)`.
///
/// Returns `Ok(None)` if the object does not implement the protocol.
fn try_zerocopy_descriptor<'py>(
    py: Python<'py>,
    obj: &PyObject,
) -> PyResult<Option<PyRef<'py, PyZeroCopyDescriptor>>> {
    let bound = obj.bind(py);
    let zc_method = match bound.getattr("__zerocopy__") {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if !zc_method.is_callable() {
        return Ok(None);
    }
    let descriptor = zc_method.call1((py.None(),))?;
    if !descriptor.is_instance_of::<PyZeroCopyDescriptor>() {
        return Err(PyValueError::new_err(
            "__zerocopy__ must return ZeroCopyDescriptor",
        ));
    }
    Ok(Some(descriptor.extract()?))
}

/// Parse single-message zerocopy payload: [4-byte header_len LE] ++ header ++ raw_data.
fn parse_zerocopy_single(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    if data.len() < 4 {
        return Err(PyValueError::new_err("Zerocopy payload too short"));
    }
    let header_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    if data.len() < 4 + header_len {
        return Err(PyValueError::new_err("Zerocopy payload truncated"));
    }
    let header: ZeroCopyDescriptorHeader =
        bincode::deserialize(&data[4..4 + header_len]).map_err(to_py_value_err)?;
    let mut offset = 4 + header_len;
    let raw_buffers: Vec<Vec<u8>> = header
        .buffer_lengths
        .iter()
        .map(|&len| {
            let buf = data[offset..offset + len].to_vec();
            offset += len;
            buf
        })
        .collect();
    let desc = PyZeroCopyDescriptor::from_wire(py, header, raw_buffers);
    let obj = Py::new(py, desc)?;
    Ok(obj.into_pyobject(py)?.into_any().unbind())
}

fn zerocopy_mode() -> String {
    std::env::var("PULSING_ZEROCOPY")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
}

/// Build a `Message::Stream` for a large zerocopy payload: descriptor header + data chunks.
fn encode_zerocopy_stream(py: Python<'_>, zc: &PyZeroCopyDescriptor) -> PyResult<Message> {
    let chunk_len = zerocopy_chunk_bytes();
    let header = zc.to_header(py);
    let header_bytes = bincode::serialize(&header).map_err(to_py_value_err)?;

    let (tx, rx) = mpsc::channel::<pulsing_actor::error::Result<Message>>(32);

    // Collect buffer data now (we hold the GIL) to avoid crossing thread boundary with PyObject
    let buffer_data: Vec<Vec<u8>> = zc
        .buffers
        .iter()
        .map(|buf_obj| {
            let bound = buf_obj.bind(py);
            let data = bound.extract::<&[u8]>()?;
            Ok(data.to_vec())
        })
        .collect::<PyResult<Vec<_>>>()?;

    std::thread::spawn(move || {
        if tx
            .blocking_send(Ok(Message::single(ZC_DESCRIPTOR_MSG_TYPE, header_bytes)))
            .is_err()
        {
            return;
        }
        for buf in &buffer_data {
            let mut offset = 0;
            while offset < buf.len() {
                let end = min(offset + chunk_len, buf.len());
                let chunk = buf[offset..end].to_vec();
                if tx
                    .blocking_send(Ok(Message::single(ZC_CHUNK_MSG_TYPE, chunk)))
                    .is_err()
                {
                    return;
                }
                offset = end;
            }
        }
    });

    Ok(Message::from_channel(ZC_DESCRIPTOR_MSG_TYPE, rx))
}

/// Reassemble a zerocopy stream (descriptor header already parsed).
/// Reads remaining data chunks from the stream and fills pre-allocated buffers.
async fn reassemble_zerocopy_stream(
    header: ZeroCopyDescriptorHeader,
    stream: &mut std::pin::Pin<
        Box<dyn futures::Stream<Item = pulsing_actor::error::Result<Message>> + Send>,
    >,
) -> pulsing_actor::error::Result<(ZeroCopyDescriptorHeader, Vec<Vec<u8>>)> {
    let mut raw_buffers: Vec<Vec<u8>> = header
        .buffer_lengths
        .iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();
    let total_expected: usize = header.buffer_lengths.iter().sum();

    let mut buf_idx = 0;
    let mut received = 0usize;

    while received < total_expected {
        let frame = stream.next().await.ok_or_else(|| {
            pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                "Zerocopy stream ended before all data received".into(),
            ))
        })??;

        match frame {
            Message::Single {
                ref msg_type,
                ref data,
            } if msg_type == ZC_CHUNK_MSG_TYPE => {
                let remaining_in_buf = header.buffer_lengths[buf_idx] - raw_buffers[buf_idx].len();
                if data.len() <= remaining_in_buf {
                    raw_buffers[buf_idx].extend_from_slice(data);
                } else {
                    // Chunk spans buffer boundary: split across buffers
                    let first_part = &data[..remaining_in_buf];
                    raw_buffers[buf_idx].extend_from_slice(first_part);
                    let mut rest = &data[remaining_in_buf..];
                    buf_idx += 1;
                    while !rest.is_empty() && buf_idx < raw_buffers.len() {
                        let can_take = min(
                            rest.len(),
                            header.buffer_lengths[buf_idx] - raw_buffers[buf_idx].len(),
                        );
                        raw_buffers[buf_idx].extend_from_slice(&rest[..can_take]);
                        rest = &rest[can_take..];
                        if raw_buffers[buf_idx].len() == header.buffer_lengths[buf_idx] {
                            buf_idx += 1;
                        }
                    }
                }
                received += data.len();
                if buf_idx < raw_buffers.len()
                    && raw_buffers[buf_idx].len() == header.buffer_lengths[buf_idx]
                {
                    buf_idx += 1;
                }
            }
            _ => {
                return Err(pulsing_actor::error::PulsingError::from(
                    pulsing_actor::error::RuntimeError::Other(format!(
                        "Unexpected frame in zerocopy stream: {:?}",
                        frame.msg_type()
                    )),
                ));
            }
        }
    }

    Ok((header, raw_buffers))
}

/// Encode a Python object into a `Message`.
///
/// Small zerocopy payloads → `Message::Single`; large ones → `Message::Stream`
/// (descriptor-first + chunked data). Non-zerocopy objects → pickle.
pub(crate) fn encode_python_payload(py: Python<'_>, obj: &PyObject) -> PyResult<Message> {
    match zerocopy_mode().as_str() {
        "off" => Ok(Message::single(SEALED_PY_MSG_TYPE, pickle_object(py, obj)?)),
        "force" => {
            let zc = try_zerocopy_descriptor(py, obj)?.ok_or_else(|| {
                PyValueError::new_err(
                    "PULSING_ZEROCOPY=force but object does not provide __zerocopy__",
                )
            })?;
            encode_zerocopy_message(py, &zc)
        }
        _ => match try_zerocopy_descriptor(py, obj)? {
            Some(zc) => encode_zerocopy_message(py, &zc),
            None => Ok(Message::single(SEALED_PY_MSG_TYPE, pickle_object(py, obj)?)),
        },
    }
}

/// Decide between single-message or stream encoding based on total buffer size.
fn encode_zerocopy_message(
    py: Python<'_>,
    zc: &PyRef<'_, PyZeroCopyDescriptor>,
) -> PyResult<Message> {
    let total = zc.total_buffer_bytes(py);
    if total >= zerocopy_stream_threshold() {
        encode_zerocopy_stream(py, zc)
    } else {
        let bytes = zc.serialize_single(py)?;
        Ok(Message::single(SEALED_ZEROCOPY_MSG_TYPE, bytes))
    }
}

/// Unified decoder: converts any `Message` (pickle / zerocopy-single / zerocopy-stream / other)
/// into a Python object.
pub(crate) async fn decode_message_to_pyobject(msg: Message) -> PyResult<PyObject> {
    match msg {
        Message::Single {
            ref msg_type,
            ref data,
        } if msg_type == SEALED_PY_MSG_TYPE => Python::with_gil(|py| unpickle_object(py, data)),
        Message::Single {
            ref msg_type,
            ref data,
        } if msg_type == SEALED_ZEROCOPY_MSG_TYPE => {
            Python::with_gil(|py| parse_zerocopy_single(py, data))
        }
        Message::Stream {
            ref default_msg_type,
            ..
        } if default_msg_type == ZC_DESCRIPTOR_MSG_TYPE => {
            let Message::Stream { mut stream, .. } = msg else {
                unreachable!()
            };
            let first = stream
                .next()
                .await
                .ok_or_else(|| PyRuntimeError::new_err("Empty zerocopy stream"))?
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let header_data = match first {
                Message::Single {
                    ref msg_type,
                    ref data,
                } if msg_type == ZC_DESCRIPTOR_MSG_TYPE => data.clone(),
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "First frame of zerocopy stream must be descriptor",
                    ));
                }
            };
            let header: ZeroCopyDescriptorHeader =
                bincode::deserialize(&header_data).map_err(to_py_value_err)?;
            let (header, raw_buffers) = reassemble_zerocopy_stream(header, &mut stream)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Python::with_gil(|py| {
                let desc = PyZeroCopyDescriptor::from_wire(py, header, raw_buffers);
                let obj = Py::new(py, desc)?;
                Ok(obj.into_pyobject(py)?.into_any().unbind())
            })
        }
        _ => Python::with_gil(|py| {
            Ok(PyMessage::from_rust_message(msg)
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }),
    }
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
                    Some(Ok(msg)) => Python::with_gil(|py| match &msg {
                        Message::Single { msg_type, data } if msg_type == SEALED_PY_MSG_TYPE => {
                            unpickle_object(py, data)
                        }
                        Message::Single { msg_type, data }
                            if msg_type == SEALED_ZEROCOPY_MSG_TYPE =>
                        {
                            parse_zerocopy_single(py, data)
                        }
                        _ => Ok(PyMessage::from_rust_message(msg)
                            .into_pyobject(py)?
                            .into_any()
                            .unbind()),
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
    sender: Arc<TokioMutex<Option<mpsc::Sender<pulsing_actor::error::Result<Message>>>>>,
}

#[pymethods]
impl PyStreamWriter {
    /// Write any Python object to the stream (auto pickle)
    ///
    /// This is the recommended method for Python-to-Python streaming.
    /// Objects are automatically pickled and will be unpickled on the reader side.
    fn write<'py>(&self, py: Python<'py>, obj: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let msg = encode_python_payload(py, &obj)?;
        let sender = self.sender.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = sender.lock().await;
            if let Some(ref tx) = *guard {
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
                let _ = tx
                    .send(Err(pulsing_actor::error::PulsingError::from(
                        pulsing_actor::error::RuntimeError::Other(msg),
                    )))
                    .await;
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
    receiver: Arc<StdMutex<Option<mpsc::Receiver<pulsing_actor::error::Result<Message>>>>>,
}

#[pymethods]
impl PyStreamMessage {
    /// Create a new streaming message with a writer.
    ///
    /// The `msg_type` is the default message type for chunks that don't specify their own.
    #[staticmethod]
    #[pyo3(signature = (msg_type, buffer_size=32))]
    fn create(msg_type: String, buffer_size: usize) -> (PyStreamMessage, PyStreamWriter) {
        let (tx, rx) = mpsc::channel::<pulsing_actor::error::Result<Message>>(buffer_size);
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
    StreamChannel(
        String,
        mpsc::Receiver<pulsing_actor::error::Result<Message>>,
    ),
    /// Pre-encoded Message (pickle single, zerocopy single, or zerocopy stream)
    Encoded(Message),
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
            encode_python_payload(py, &msg)?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = actor_ref.send(actor_msg).await.map_err(to_pyerr)?;
            decode_message_to_pyobject(response).await
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
            encode_python_payload(py, &msg)?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            actor_ref.send_oneway(actor_msg).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Return an untyped proxy that forwards any method call to the remote actor.
    fn as_any(&self, py: Python<'_>) -> PyResult<PyObject> {
        let remote = py.import("pulsing.core.remote")?;
        let proxy_cls = remote.getattr("ActorProxy")?;
        let proxy = proxy_cls.call1((self.clone(), py.None(), py.None()))?;
        Ok(proxy.unbind())
    }

    /// Return a typed proxy based on the given class definition.
    fn as_type(&self, py: Python<'_>, cls: PyObject) -> PyResult<PyObject> {
        let remote = py.import("pulsing.core.remote")?;
        let extract_fn = remote.getattr("_extract_methods")?;
        let result = extract_fn.call1((&cls,))?;
        let methods = result.get_item(0)?;
        let async_methods = result.get_item(1)?;
        let proxy_cls = remote.getattr("ActorProxy")?;
        let proxy = proxy_cls.call1((self.clone(), methods, async_methods))?;
        Ok(proxy.unbind())
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
        let socket_addr: SocketAddr = addr.parse().map_err(to_py_value_err)?;
        Ok(Self {
            inner: SystemConfig::with_addr(socket_addr),
        })
    }

    fn with_seeds(&self, seeds: Vec<String>) -> PyResult<Self> {
        let seed_addrs: Result<Vec<SocketAddr>, _> = seeds.iter().map(|s| s.parse()).collect();
        let seed_addrs = seed_addrs.map_err(to_py_value_err)?;
        Ok(Self {
            inner: self.inner.clone().with_seeds(seed_addrs),
        })
    }

    /// Run this node as the head node (workers will register with it).
    fn with_head_node(&self) -> Self {
        Self {
            inner: self.inner.clone().with_head_node(),
        }
    }

    /// Connect to a head node at the given address (makes this node a worker).
    fn with_head_addr(&self, addr: String) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr.parse().map_err(to_py_value_err)?;
        Ok(Self {
            inner: self.inner.clone().with_head_addr(socket_addr),
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

    async fn on_start(&mut self, ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let actor_id = *ctx.id();
        let event_loop = Python::with_gil(|py| self.event_loop.clone_ref(py));

        python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<()> {
                    if handler.getattr(py, "on_start").is_ok() {
                        let py_actor_id = PyActorId { inner: actor_id };
                        let result = handler.call_method1(py, "on_start", (py_actor_id,))?;

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
            .map_err(|e| {
                pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                    format!("Python executor error: {:?}", e),
                ))
            })?
            .map_err(|e| {
                pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                    format!("Python on_start error: {:?}", e),
                ))
            })
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let event_loop = Python::with_gil(|py| self.event_loop.clone_ref(py));

        python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<()> {
                    if handler.getattr(py, "on_stop").is_ok() {
                        let result = handler.call_method0(py, "on_stop")?;

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
            .map_err(|e| {
                pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                    format!("Python executor error: {:?}", e),
                ))
            })?
            .map_err(|e| {
                pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                    format!("Python on_stop error: {:?}", e),
                ))
            })
    }

    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let (handler, event_loop) =
            Python::with_gil(|py| (self.handler.clone_ref(py), self.event_loop.clone_ref(py)));

        // Decode-first: convert any message format to a Python object
        let call_arg = decode_message_to_pyobject(msg).await.map_err(|e| {
            pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                e.to_string(),
            ))
        })?;

        let response: Result<PyActorResponse, PyErr> = python_executor()
            .execute(move || {
                Python::with_gil(|py| -> PyResult<PyActorResponse> {
                    let receive_method = handler.getattr(py, "receive")?;
                    let result = match receive_method.call1(py, (&call_arg,)) {
                        Ok(value) => value,
                        Err(py_err) => return Err(py_err),
                    };

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

                    let type_name = py_result_bound
                        .get_type()
                        .qualname()
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    if type_name == "async_generator" || type_name == "generator" {
                        return Ok(PyActorResponse::Generator(
                            py_result.clone_ref(py),
                            event_loop.clone_ref(py),
                            type_name == "async_generator",
                        ));
                    }

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

                    if py_result_bound.is_instance_of::<PyMessage>() {
                        let msg: PyMessage = py_result_bound.extract()?;
                        if msg.is_stream() {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "PyMessage with stream cannot be returned from receive(), use StreamMessage instead"
                            ));
                        }
                        return Ok(PyActorResponse::Single(msg));
                    }

                    let msg = encode_python_payload(py, &py_result)?;
                    Ok(PyActorResponse::Encoded(msg))
                })
            })
            .await
            .map_err(|e| {
                pulsing_actor::error::PulsingError::from(
                    pulsing_actor::error::RuntimeError::Other(
                        format!("Python executor error: {:?}", e),
                    ),
                )
            })?;

        let response = match response {
            Ok(resp) => resp,
            Err(py_err) => {
                Python::with_gil(|py| {
                    let actor_err =
                        convert_python_exception_to_actor_error(py, &py_err).map_err(|e| {
                            pulsing_actor::error::PulsingError::from(
                                pulsing_actor::error::RuntimeError::Other(e.to_string()),
                            )
                        })?;
                    Err(pulsing_actor::error::PulsingError::from(actor_err))
                })
            }?,
        };

        match response {
            PyActorResponse::Single(msg) => Ok(msg.to_message()),
            PyActorResponse::StreamChannel(default_msg_type, rx) => {
                Ok(Message::from_channel(&default_msg_type, rx))
            }
            PyActorResponse::Encoded(msg) => Ok(msg),
            PyActorResponse::Generator(generator, event_loop, is_async) => {
                let (tx, rx) = mpsc::channel::<pulsing_actor::error::Result<Message>>(32);
                tokio::spawn(async move {
                    let result = python_executor()
                        .execute(move || {
                            Python::with_gil(|py| -> PyResult<()> {
                                let gen = generator.bind(py);
                                let asyncio = py.import("asyncio")?;
                                if is_async {
                                    let run_coroutine_threadsafe =
                                        asyncio.getattr("run_coroutine_threadsafe")?;
                                    loop {
                                        let anext_coro = gen.call_method0("__anext__")?;
                                        let future = run_coroutine_threadsafe
                                            .call1((&anext_coro, &event_loop))?;
                                        match future.call_method0("result") {
                                            Ok(item) => {
                                                let item_obj = item.unbind();
                                                let msg = encode_python_payload(py, &item_obj)?;
                                                if tx.blocking_send(Ok(msg)).is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                                                    break;
                                                }
                                                let _ = tx.blocking_send(Err(
                                                    pulsing_actor::error::PulsingError::from(
                                                        pulsing_actor::error::RuntimeError::Other(
                                                            format!("Generator error: {}", e),
                                                        ),
                                                    ),
                                                ));
                                                break;
                                            }
                                        }
                                    }
                                } else {
                                    loop {
                                        match gen.call_method0("__next__") {
                                            Ok(item) => {
                                                let item_obj = item.unbind();
                                                let msg = encode_python_payload(py, &item_obj)?;
                                                if tx.blocking_send(Ok(msg)).is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                                    break;
                                                }
                                                let _ = tx.blocking_send(Err(
                                                    pulsing_actor::error::PulsingError::from(
                                                        pulsing_actor::error::RuntimeError::Other(
                                                            format!("Generator error: {}", e),
                                                        ),
                                                    ),
                                                ));
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
            let options = pulsing_actor::system::SpawnOptions::default()
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
                        .spawning()
                        .metadata(options.metadata)
                        .spawn(actor_wrapper)
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
                        ActorPath::new_system(&name).map_err(to_py_value_err)?
                    } else {
                        ActorPath::new(&name).map_err(to_py_value_err)?
                    };

                    if matches!(policy, RestartPolicy::Never) {
                        // actor is the instance
                        let actor_wrapper = PythonActorWrapper::new(actor, event_loop);
                        system
                            .spawning()
                            .path(path)
                            .supervision(options.supervision)
                            .metadata(options.metadata)
                            .spawn(actor_wrapper)
                            .await
                            .map_err(to_pyerr)?
                    } else {
                        // actor is a factory - named actor with supervision
                        let factory = move || {
                            Python::with_gil(
                                |py| -> pulsing_actor::error::Result<PythonActorWrapper> {
                                    let event_loop = event_loop.clone_ref(py);
                                    let instance = actor.call0(py).map_err(|e| {
                                        pulsing_actor::error::PulsingError::from(
                                            pulsing_actor::error::RuntimeError::Other(format!(
                                                "Python factory error: {:?}",
                                                e
                                            )),
                                        )
                                    })?;
                                    Ok(PythonActorWrapper::new(instance, event_loop))
                                },
                            )
                        };
                        system
                            .spawning()
                            .path(path)
                            .supervision(options.supervision)
                            .metadata(options.metadata)
                            .spawn_factory(factory)
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
            Python::with_gil(|py| -> PyResult<PyObject> {
                use pyo3::types::PyDict;
                let list = pyo3::types::PyList::empty(py);
                for m in members {
                    let dict = PyDict::new(py);
                    dict.set_item("node_id", m.node_id.0)?;
                    dict.set_item("addr", m.addr.to_string())?;
                    dict.set_item("status", format!("{:?}", m.status))?;
                    list.append(dict)?;
                }
                Ok(list.into_pyobject(py)?.into())
            })
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
                ActorPath::new_system(&name).map_err(to_py_value_err)?
            } else {
                ActorPath::new(&name).map_err(to_py_value_err)?
            };
            let instances = system.get_named_instances_detailed(&path).await;

            Python::with_gil(|py| -> PyResult<PyObject> {
                use pyo3::types::PyDict;
                let list = pyo3::types::PyList::empty(py);
                for (member, instance_opt) in instances {
                    let dict = PyDict::new(py);
                    dict.set_item("node_id", member.node_id.0)?;
                    dict.set_item("addr", member.addr.to_string())?;
                    dict.set_item("status", format!("{:?}", member.status))?;
                    if let Some(inst) = instance_opt {
                        dict.set_item("actor_id", inst.actor_id.0)?;
                        for (k, v) in inst.metadata {
                            dict.set_item(k, v)?;
                        }
                    }
                    list.append(dict)?;
                }
                Ok(list.into_pyobject(py)?.into())
            })
        })
    }

    /// Get all named actors in the cluster
    fn all_named_actors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let all_named = system.all_named_actors().await;

            Python::with_gil(|py| -> PyResult<PyObject> {
                use pyo3::types::PyDict;
                let list = pyo3::types::PyList::empty(py);
                for info in all_named {
                    let dict = PyDict::new(py);
                    dict.set_item("path", info.path.as_str())?;
                    dict.set_item("instance_count", info.instance_count())?;
                    let instances_list = pyo3::types::PyList::empty(py);
                    for id in &info.instance_nodes {
                        instances_list.append(id.0)?;
                    }
                    dict.set_item("instances", instances_list)?;
                    if !info.instances.is_empty() {
                        let detailed_list = pyo3::types::PyList::empty(py);
                        for (node_id, inst) in &info.instances {
                            let inst_dict = PyDict::new(py);
                            inst_dict.set_item("node_id", node_id.0)?;
                            inst_dict.set_item("actor_id", inst.actor_id.0)?;
                            for (k, v) in &inst.metadata {
                                inst_dict.set_item(k, v)?;
                            }
                            detailed_list.append(inst_dict)?;
                        }
                        dict.set_item("detailed_instances", detailed_list)?;
                    }
                    list.append(dict)?;
                }
                Ok(list.into_pyobject(py)?.into())
            })
        })
    }

    /// Resolve a named actor (selects one instance using load balancing)
    ///
    /// When `timeout` is provided, retries resolution until the name appears
    /// or the timeout expires (useful for waiting on gossip propagation).
    #[pyo3(signature = (name, node_id=None, timeout=None))]
    fn resolve_named<'py>(
        &self,
        py: Python<'py>,
        name: String,
        node_id: Option<u128>,
        timeout: Option<f64>,
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
                ActorPath::new_system(&name).map_err(to_py_value_err)?
            } else {
                ActorPath::new(&name).map_err(to_py_value_err)?
            };
            let node = node_id.map(NodeId::new);

            match timeout {
                None => {
                    // No timeout: error immediately if not found (original behavior)
                    let actor_ref = system
                        .resolve_named(&path, node.as_ref())
                        .await
                        .map_err(to_pyerr)?;
                    Ok(PyActorRef { inner: actor_ref })
                }
                Some(secs) => {
                    // With timeout: retry until name appears or timeout
                    let deadline =
                        tokio::time::Instant::now() + std::time::Duration::from_secs_f64(secs);
                    let mut last_err = None;
                    while tokio::time::Instant::now() < deadline {
                        match system.resolve_named(&path, node.as_ref()).await {
                            Ok(actor_ref) => return Ok(PyActorRef { inner: actor_ref }),
                            Err(e) => last_err = Some(e),
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    }
                    Err(to_pyerr(last_err.unwrap()))
                }
            }
        })
    }

    /// Alias for resolve_named - resolve actor by name
    #[pyo3(signature = (name, *, node_id=None, timeout=None))]
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        name: String,
        node_id: Option<u128>,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.resolve_named(py, name, node_id, timeout)
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
    fn remote_system<'py>(&self, py: Python<'py>, node_id: u128) -> PyResult<Bound<'py, PyAny>> {
        let system = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Use system/core - the correct system actor path
            let path = ActorPath::new_system("system/core").map_err(to_py_value_err)?;
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
    m.add_class::<PyZeroCopyDescriptor>()?;
    Ok(())
}
