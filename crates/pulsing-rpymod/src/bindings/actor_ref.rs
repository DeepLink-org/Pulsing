use pulsing_actor::prelude::{ActorRef, Message};
use pulsing_actor::tracing::{
    capture_linked_traceparent_for_mailbox, capture_linked_tracestate_for_mailbox,
};
use rustpython_vm::{AsObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use super::codec::{decode_message_to_pyobject, py_message_to_rust};
use super::message::PyActorId;
use crate::interop::{current_event_loop, defer_async, read_contextvar, IntoPyResult};

#[pyclass(module = "pulsing._core", name = "ActorRef")]
#[derive(Clone, Debug, PyPayload)]
pub struct PyActorRef {
    pub(crate) inner: ActorRef,
}

#[pyclass]
impl PyActorRef {
    #[pygetset]
    fn actor_id(&self) -> PyActorId {
        PyActorId {
            inner: *self.inner.id(),
        }
    }

    #[pymethod]
    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    #[pymethod]
    fn ask(&self, msg: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let actor_ref = self.inner.clone();
        let actor_msg = py_message_to_rust(vm, &msg)?;
        let tp = read_contextvar(vm, "_current_traceparent")
            .or_else(capture_linked_traceparent_for_mailbox);
        let ts = read_contextvar(vm, "_current_tracestate")
            .or_else(capture_linked_tracestate_for_mailbox);
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let response = actor_ref
                .send_with_trace(actor_msg, tp, ts)
                .await
                .map_err(|e| e.to_string())?;
            Ok(response)
        })
    }

    #[pymethod]
    fn tell(&self, msg: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let actor_ref = self.inner.clone();
        let actor_msg = py_message_to_rust(vm, &msg)?;
        let tp = read_contextvar(vm, "_current_traceparent")
            .or_else(capture_linked_traceparent_for_mailbox);
        let ts = read_contextvar(vm, "_current_tracestate")
            .or_else(capture_linked_tracestate_for_mailbox);
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            actor_ref
                .send_oneway_with_trace(actor_msg, tp, ts)
                .await
                .map_err(|e| e.to_string())?;
            Ok(())
        })
    }

    #[pymethod]
    fn as_any(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let remote = vm.import("pulsing.core.remote", 0)?;
        let proxy_cls = remote.get_attr("ActorProxy", vm)?;
        let me: PyObjectRef = self.clone().into_ref(&vm.ctx).into();
        proxy_cls.call((me, vm.ctx.none(), vm.ctx.none()), vm)
    }

    #[pymethod]
    fn as_type(&self, cls: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let remote = vm.import("pulsing.core.remote", 0)?;
        let extract_fn = remote.get_attr("_extract_methods", vm)?;
        let result = extract_fn.call((cls,), vm)?;
        let methods = result
            .get_attr("0", vm)
            .or_else(|_| result.get_item(vm.ctx.new_int(0).as_object(), vm))?;
        let async_methods = result
            .get_attr("1", vm)
            .or_else(|_| result.get_item(vm.ctx.new_int(1).as_object(), vm))?;
        let proxy_cls = remote.get_attr("ActorProxy", vm)?;
        let me: PyObjectRef = self.clone().into_ref(&vm.ctx).into();
        proxy_cls.call((me, methods, async_methods), vm)
    }
}

impl IntoPyResult for Message {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let rt = crate::runtime::tokio_handle().map_err(|e| vm.new_runtime_error(e.to_string()))?;
        rt.block_on(decode_message_to_pyobject(vm, self))
    }
}
