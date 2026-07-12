use futures::StreamExt;
use pulsing_actor::prelude::Message;
use rustpython_vm::{AsObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use super::codec::{decode_message_to_pyobject, encode_python_payload};
use super::message::{PyStreamReader, PyStreamWriter};
use crate::interop::{current_event_loop, defer_async, IntoPyResult};

#[pyclass]
impl PyStreamReader {
    #[pymethod(name = "__aiter__")]
    fn aiter(zelf: PyRef<Self>) -> PyRef<Self> {
        zelf
    }

    #[pymethod]
    fn __anext__(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let stream = self.stream.clone();
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let mut guard = stream.lock().await;
            if let Some(ref mut s) = *guard {
                match s.next().await {
                    Some(Ok(msg)) => Ok(AnextOutcome::Message(msg)),
                    Some(Err(e)) => Err(e.to_string()),
                    None => {
                        *guard = None;
                        Ok(AnextOutcome::Stop)
                    }
                }
            } else {
                Ok(AnextOutcome::Stop)
            }
        })
    }

    #[pymethod]
    fn cancel(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let stream = self.stream.clone();
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let mut guard = stream.lock().await;
            *guard = None;
            Ok(())
        })
    }
}

#[pyclass]
impl PyStreamWriter {
    #[pymethod]
    fn write(&self, obj: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let msg = encode_python_payload(vm, &obj)?;
        let sender = self.sender.clone();
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let guard = sender.lock().await;
            if let Some(ref tx) = *guard {
                tx.send(Ok(msg))
                    .await
                    .map_err(|_| "Stream closed".to_string())?;
                Ok(())
            } else {
                Err("Writer already closed".to_string())
            }
        })
    }

    #[pymethod]
    fn close(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let sender = self.sender.clone();
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let mut guard = sender.lock().await;
            *guard = None;
            Ok(())
        })
    }

    #[pymethod]
    fn error(&self, msg: rustpython_vm::builtins::PyUtf8StrRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let sender = self.sender.clone();
        let err = msg.as_str().to_string();
        let event_loop = current_event_loop(vm)?;
        defer_async(vm, event_loop, async move {
            let mut guard = sender.lock().await;
            if let Some(tx) = guard.take() {
                let _ = tx
                    .send(Err(pulsing_actor::error::PulsingError::from(
                        pulsing_actor::error::RuntimeError::Other(err),
                    )))
                    .await;
            }
            Ok(())
        })
    }
}

enum AnextOutcome {
    Message(Message),
    Stop,
}

impl IntoPyResult for AnextOutcome {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        match self {
            AnextOutcome::Stop => Err(vm.new_exception(
                vm.ctx.exceptions.stop_async_iteration.to_owned(),
                vec![],
            )),
            AnextOutcome::Message(msg) => {
                let rt = crate::runtime::tokio_handle()
                    .map_err(|e| vm.new_runtime_error(e.to_string()))?;
                rt.block_on(decode_message_to_pyobject(vm, msg))
            }
        }
    }
}
