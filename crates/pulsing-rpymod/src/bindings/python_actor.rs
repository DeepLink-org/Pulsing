use async_trait::async_trait;
use pulsing_actor::actor::ActorContext;
use pulsing_actor::prelude::{Actor, Message};
use rustpython_vm::{PyObjectRef, PyResult, VirtualMachine};

use super::codec::{decode_message_to_pyobject, encode_python_payload};
use super::message::{PyMessage, PyStreamMessage};
use crate::interop::{is_coroutine, schedule_vm_work};
use crate::send_py::SendPy;

pub struct PythonActorWrapper {
    handler: PyObjectRef,
    event_loop: PyObjectRef,
}

unsafe impl Send for PythonActorWrapper {}
unsafe impl Sync for PythonActorWrapper {}

impl PythonActorWrapper {
    pub fn new(handler: PyObjectRef, event_loop: PyObjectRef) -> Self {
        Self {
            handler,
            event_loop,
        }
    }
}

#[async_trait]
impl Actor for PythonActorWrapper {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let handler = SendPy::new(self.handler.clone());
        let event_loop = SendPy::new(self.event_loop.clone());
        let (tx, rx) = tokio::sync::oneshot::channel::<pulsing_actor::error::Result<Message>>();
        schedule_vm_work(Box::new(move |vm| {
            let handler = handler.into_inner();
            let event_loop = event_loop.into_inner();
            let result = (|| -> PyResult<Message> {
                let rt = crate::runtime::tokio_handle()
                    .map_err(|e| vm.new_runtime_error(e.to_string()))?;
                let py_arg = rt.block_on(decode_message_to_pyobject(vm, msg))?;
                let receive = handler.get_attr("receive", vm)?;
                let result = receive.call((py_arg,), vm)?;
                if is_coroutine(vm, &result).unwrap_or(false) {
                    return crate::interop::await_coroutine_on_running_loop(
                        vm,
                        &event_loop,
                        result,
                    )
                    .and_then(|py_result| encode_response(vm, py_result));
                }
                encode_response(vm, result)
            })();
            let rust_result = result.map_err(|_| {
                pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                    "Python actor receive failed".into(),
                ))
            });
            let _ = tx.send(rust_result);
        }));
        rx.await.map_err(|_| {
            pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                "Python receive channel closed".into(),
            ))
        })?
    }
}

fn encode_response(vm: &VirtualMachine, py_result: PyObjectRef) -> PyResult<Message> {
    if vm.is_none(&py_result) {
        return Ok(PyMessage::empty_msg().to_message());
    }
    if let Some(stream_msg) = py_result.downcast_ref::<PyStreamMessage>() {
        let default_msg_type = stream_msg.default_msg_type.clone();
        let receiver = stream_msg
            .receiver
            .lock()
            .map_err(|e| vm.new_runtime_error(e.to_string()))?
            .take()
            .ok_or_else(|| vm.new_runtime_error("StreamMessage receiver already consumed"))?;
        return Ok(Message::from_channel(&default_msg_type, receiver));
    }
    if let Some(py_msg) = py_result.downcast_ref::<PyMessage>() {
        return Ok(py_msg.to_message());
    }
    encode_python_payload(vm, &py_result)
}
