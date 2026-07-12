use std::net::SocketAddr;
use std::sync::Arc;

use pulsing_actor::actor::{ActorId, NodeId};
use pulsing_actor::prelude::{Message, SystemConfig};
use pulsing_bindings_core::{parse_actor_id, PyActorIdView, PyNodeIdView};
use rustpython_vm::builtins::{PyBytes, PyBytesRef, PyUtf8StrRef};
use rustpython_vm::function::OptionalArg;
use rustpython_vm::types::Constructor;
use rustpython_vm::{AsObject, FromArgs, Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};
use tokio::sync::Mutex as TokioMutex;

use super::codec::ensure_contiguous_buffer;

#[pyclass(module = "pulsing._core", name = "NodeId")]
#[derive(Clone, Debug, PyPayload)]
pub struct PyNodeId {
    pub(crate) inner: NodeId,
}

#[pyclass]
impl PyNodeId {
    #[pystaticmethod]
    fn generate() -> Self {
        Self {
            inner: PyNodeIdView::generate().0,
        }
    }

    #[pystaticmethod]
    fn local() -> Self {
        Self {
            inner: PyNodeIdView::local().0,
        }
    }

    #[pygetset]
    fn id(&self) -> u128 {
        PyNodeIdView(self.inner).id()
    }

    #[pymethod]
    fn uuid(&self) -> String {
        PyNodeIdView(self.inner).uuid()
    }

    #[pymethod]
    fn is_local(&self) -> bool {
        PyNodeIdView(self.inner).is_local()
    }

    #[pymethod(name = "__str__")]
    fn str_repr(&self) -> String {
        self.inner.to_string()
    }
}

#[pyclass(module = "pulsing._core", name = "ActorId")]
#[derive(Clone, Debug, PyPayload)]
pub struct PyActorId {
    pub(crate) inner: ActorId,
}

#[pyclass]
impl PyActorId {
    #[pystaticmethod]
    fn generate() -> Self {
        Self {
            inner: PyActorIdView::generate().0,
        }
    }

    #[pystaticmethod]
    fn from_str(s: PyUtf8StrRef, vm: &VirtualMachine) -> PyResult<Self> {
        let inner = parse_actor_id(Some(s.as_str()), None).map_err(|e| vm.new_value_error(e))?;
        Ok(Self { inner })
    }

    #[pygetset]
    fn id(&self) -> u128 {
        PyActorIdView(self.inner).id()
    }

    #[pymethod]
    fn uuid(&self) -> String {
        PyActorIdView(self.inner).uuid()
    }

    #[pymethod(name = "__str__")]
    fn str_repr(&self) -> String {
        self.inner.to_string()
    }
}

#[derive(FromArgs)]
pub struct PyMessageNewArgs {
    msg_type: PyUtf8StrRef,
    #[pyarg(positional, optional)]
    payload: Option<PyBytesRef>,
}

#[pyclass(module = "pulsing._core", name = "Message")]
#[derive(Clone, PyPayload)]
pub struct PyMessage {
    pub(crate) msg_type: String,
    pub(crate) payload: Option<Vec<u8>>,
    pub(crate) stream_reader: Option<Arc<TokioMutex<Option<pulsing_actor::actor::MessageStream>>>>,
}

impl std::fmt::Debug for PyMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyMessage")
            .field("msg_type", &self.msg_type)
            .field("is_stream", &self.stream_reader.is_some())
            .finish()
    }
}

impl Constructor for PyMessage {
    type Args = PyMessageNewArgs;

    fn py_new(
        _cls: &Py<rustpython_vm::builtins::PyType>,
        args: Self::Args,
        _vm: &VirtualMachine,
    ) -> PyResult<Self> {
        Ok(Self {
            msg_type: args.msg_type.as_str().to_string(),
            payload: Some(
                args.payload
                    .map(|b| b.as_bytes().to_vec())
                    .unwrap_or_default(),
            ),
            stream_reader: None,
        })
    }
}

#[pyclass(with(Constructor))]
impl PyMessage {
    #[pystaticmethod]
    fn from_json(msg_type: PyUtf8StrRef, data: PyObjectRef, vm: &VirtualMachine) -> PyResult<Self> {
        let json = vm.import("json", 0)?;
        let dumped = vm.call_method(json.as_object(), "dumps", (data,))?;
        let s = dumped.try_into_value::<PyUtf8StrRef>(vm)?;
        Ok(Self {
            msg_type: msg_type.as_str().to_string(),
            payload: Some(s.as_str().as_bytes().to_vec()),
            stream_reader: None,
        })
    }

    #[pystaticmethod]
    fn empty() -> Self {
        Self::empty_msg()
    }

    #[pygetset]
    fn msg_type(&self) -> String {
        self.msg_type.clone()
    }

    #[pygetset]
    fn is_stream(&self) -> bool {
        self.stream_reader.is_some()
    }

    #[pygetset]
    fn payload(&self, vm: &VirtualMachine) -> PyResult<PyBytesRef> {
        match &self.payload {
            Some(data) => Ok(vm.ctx.new_bytes(data.clone())),
            None => Err(vm.new_value_error(
                "Cannot get payload from stream message, use stream_reader() instead",
            )),
        }
    }

    #[pymethod]
    fn to_json(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.payload.as_ref().ok_or_else(|| {
            vm.new_value_error("Cannot parse stream message as JSON, use stream_reader() instead")
        })?;
        let json = vm.import("json", 0)?;
        let s = vm.ctx.new_str(String::from_utf8_lossy(data).into_owned());
        vm.call_method(json.as_object(), "loads", (s,))
    }

    #[pymethod]
    fn stream_reader(&self, vm: &VirtualMachine) -> PyResult<PyRef<PyStreamReader>> {
        match &self.stream_reader {
            Some(stream) => Ok(PyStreamReader {
                stream: stream.clone(),
            }
            .into_ref(&vm.ctx)),
            None => Err(vm.new_value_error(
                "This is not a stream message, access payload directly",
            )),
        }
    }
}

impl PyMessage {
    pub(crate) fn empty_msg() -> Self {
        Self {
            msg_type: String::new(),
            payload: Some(Vec::new()),
            stream_reader: None,
        }
    }

    pub(crate) fn to_message(&self) -> Message {
        if self.stream_reader.is_some() {
            Message::single(&self.msg_type, Vec::new())
        } else {
            Message::single(&self.msg_type, self.payload.clone().unwrap_or_default())
        }
    }

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

#[derive(FromArgs)]
pub struct PyZeroCopyDescriptorNewArgs {
    buffers: Vec<PyObjectRef>,
    #[pyarg(named, optional)]
    dtype: Option<PyUtf8StrRef>,
    #[pyarg(named, optional)]
    shape: Option<Vec<usize>>,
    #[pyarg(named, optional)]
    strides: Option<Vec<isize>>,
    #[pyarg(named, optional)]
    transport: Option<PyUtf8StrRef>,
    #[pyarg(named, optional)]
    checksum: Option<PyUtf8StrRef>,
    #[pyarg(named, optional)]
    version: Option<u32>,
}

#[pyclass(module = "pulsing._core", name = "ZeroCopyDescriptor")]
#[derive(Clone, Debug, PyPayload)]
pub struct PyZeroCopyDescriptor {
    pub(crate) version: u32,
    pub(crate) buffers: Vec<PyObjectRef>,
    pub(crate) dtype: Option<String>,
    pub(crate) shape: Option<Vec<usize>>,
    pub(crate) strides: Option<Vec<isize>>,
    pub(crate) transport: Option<String>,
    pub(crate) checksum: Option<String>,
}

impl Constructor for PyZeroCopyDescriptor {
    type Args = PyZeroCopyDescriptorNewArgs;

    fn py_new(
        _cls: &Py<rustpython_vm::builtins::PyType>,
        args: Self::Args,
        vm: &VirtualMachine,
    ) -> PyResult<Self> {
        if args.buffers.is_empty() {
            return Err(vm.new_value_error("ZeroCopyDescriptor requires at least one buffer"));
        }
        let normalized = args
            .buffers
            .into_iter()
            .map(|item| ensure_contiguous_buffer(vm, &item))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            version: args.version.unwrap_or(1),
            buffers: normalized,
            dtype: args.dtype.map(|s| s.as_str().to_string()),
            shape: args.shape,
            strides: args.strides,
            transport: args.transport.map(|s| s.as_str().to_string()),
            checksum: args.checksum.map(|s| s.as_str().to_string()),
        })
    }
}

#[pyclass(with(Constructor))]
impl PyZeroCopyDescriptor {
    #[pygetset]
    fn version(&self) -> u32 {
        self.version
    }

    #[pygetset]
    fn buffers(&self) -> Vec<PyObjectRef> {
        self.buffers.clone()
    }

    #[pygetset]
    fn dtype(&self, vm: &VirtualMachine) -> PyObjectRef {
        match &self.dtype {
            Some(v) => vm.ctx.new_str(v.clone()).into(),
            None => vm.ctx.none(),
        }
    }

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        match &self.shape {
            Some(v) => vm
                .ctx
                .new_list(v.iter().map(|n| vm.ctx.new_int(*n).into()).collect())
                .into(),
            None => vm.ctx.none(),
        }
    }

    #[pygetset]
    fn strides(&self, vm: &VirtualMachine) -> PyObjectRef {
        match &self.strides {
            Some(v) => vm
                .ctx
                .new_list(v.iter().map(|n| vm.ctx.new_int(*n).into()).collect())
                .into(),
            None => vm.ctx.none(),
        }
    }

    #[pygetset]
    fn transport(&self, vm: &VirtualMachine) -> PyObjectRef {
        match &self.transport {
            Some(v) => vm.ctx.new_str(v.clone()).into(),
            None => vm.ctx.none(),
        }
    }

    #[pygetset]
    fn checksum(&self, vm: &VirtualMachine) -> PyObjectRef {
        match &self.checksum {
            Some(v) => vm.ctx.new_str(v.clone()).into(),
            None => vm.ctx.none(),
        }
    }
}

#[pyclass(module = "pulsing._core", name = "StreamReader")]
#[derive(PyPayload)]
pub struct PyStreamReader {
    pub(crate) stream: Arc<TokioMutex<Option<pulsing_actor::actor::MessageStream>>>,
}

impl std::fmt::Debug for PyStreamReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyStreamReader").finish()
    }
}

#[pyclass(module = "pulsing._core", name = "StreamWriter")]
#[derive(Debug, PyPayload)]
pub struct PyStreamWriter {
    pub(crate) sender: Arc<TokioMutex<Option<tokio::sync::mpsc::Sender<pulsing_actor::error::Result<Message>>>>>,
}

#[pyclass(module = "pulsing._core", name = "StreamMessage")]
#[derive(Debug, PyPayload)]
pub struct PyStreamMessage {
    pub(crate) default_msg_type: String,
    pub(crate) receiver: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<pulsing_actor::error::Result<Message>>>>>,
}

#[pyclass]
impl PyStreamMessage {
    #[pystaticmethod]
    fn create(
        msg_type: PyUtf8StrRef,
        buffer_size: OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<(PyRef<Self>, PyRef<PyStreamWriter>)> {
        let (tx, rx) = tokio::sync::mpsc::channel(buffer_size.into_option().unwrap_or(32));
        Ok((
            Self {
                default_msg_type: msg_type.as_str().to_string(),
                receiver: Arc::new(std::sync::Mutex::new(Some(rx))),
            }
            .into_ref(&vm.ctx),
            PyStreamWriter {
                sender: Arc::new(TokioMutex::new(Some(tx))),
            }
            .into_ref(&vm.ctx),
        ))
    }

    #[pygetset]
    fn msg_type(&self) -> String {
        self.default_msg_type.clone()
    }
}

#[pyclass(module = "pulsing._core", name = "SystemConfig")]
#[derive(Clone, Debug, PyPayload)]
pub struct PySystemConfig {
    pub(crate) inner: SystemConfig,
}

#[pyclass]
impl PySystemConfig {
    #[pystaticmethod]
    fn standalone() -> Self {
        Self {
            inner: SystemConfig::standalone(),
        }
    }

    #[pystaticmethod]
    fn with_addr(addr: PyUtf8StrRef, vm: &VirtualMachine) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr
            .as_str()
            .parse()
            .map_err(|e: std::net::AddrParseError| vm.new_value_error(e.to_string()))?;
        Ok(Self {
            inner: SystemConfig::with_addr(socket_addr),
        })
    }

    #[pymethod]
    fn with_seeds(&self, seeds: Vec<PyUtf8StrRef>, vm: &VirtualMachine) -> PyResult<Self> {
        let seed_addrs: Result<Vec<SocketAddr>, std::net::AddrParseError> =
            seeds.iter().map(|s| s.as_str().parse()).collect();
        let seed_addrs = seed_addrs.map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_seeds(seed_addrs),
        })
    }

    #[pymethod]
    fn with_head_node(&self) -> Self {
        Self {
            inner: self.inner.clone().with_head_node(),
        }
    }

    #[pymethod]
    fn with_head_addr(&self, addr: PyUtf8StrRef, vm: &VirtualMachine) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr
            .as_str()
            .parse()
            .map_err(|e: std::net::AddrParseError| vm.new_value_error(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_head_addr(socket_addr),
        })
    }

    #[pymethod]
    fn is_tls_enabled(&self) -> bool {
        self.inner.is_tls_enabled()
    }
}
