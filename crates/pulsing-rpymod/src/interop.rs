//! Bridge tokio futures to asyncio and schedule Python work on the VM thread.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Mutex, OnceLock};

use rustpython_vm::builtins::PyStrRef;
use rustpython_vm::{AsObject, PyObjectRef, PyResult, VirtualMachine};

use crate::runtime;

type VmWork = Box<dyn FnOnce(&VirtualMachine) + Send>;

fn queue() -> &'static Mutex<VecDeque<VmWork>> {
    static Q: OnceLock<Mutex<VecDeque<VmWork>>> = OnceLock::new();
    Q.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub fn schedule_vm_work(work: VmWork) {
    if let Ok(mut q) = queue().lock() {
        q.push_back(work);
    }
}

pub fn drain_vm_queue(vm: &VirtualMachine) {
    loop {
        let work = queue().lock().ok().and_then(|mut q| q.pop_front());
        match work {
            Some(f) => f(vm),
            None => break,
        }
    }
}

pub fn defer_async<F, T>(
    vm: &VirtualMachine,
    event_loop: PyObjectRef,
    fut: F,
) -> PyResult<PyObjectRef>
where
    F: Future<Output = Result<T, String>> + Send + 'static,
    T: IntoPyResult + Send + 'static,
{
    use crate::send_py::SendPy;

    let py_future = vm.call_method(event_loop.as_object(), "create_future", ())?;
    let py_future_done = SendPy::new(py_future.clone());
    let handle = runtime::tokio_handle().map_err(|e| vm.new_runtime_error(e.to_string()))?;

    handle.spawn(async move {
        let result = fut.await;
        let py_future_done = py_future_done;
        schedule_vm_work(Box::new(move |vm| {
            let py_future_done = py_future_done.into_inner();
            match result {
                Ok(val) => {
                    let py_val = match val.into_pyresult(vm) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ =
                                vm.call_method(py_future_done.as_object(), "set_exception", (e,));
                            return;
                        }
                    };
                    let _ = vm.call_method(py_future_done.as_object(), "set_result", (py_val,));
                }
                Err(err) => {
                    let exc = vm.new_runtime_error(err);
                    let _ = vm.call_method(py_future_done.as_object(), "set_exception", (exc,));
                }
            }
        }));
    });

    Ok(py_future)
}

pub trait IntoPyResult {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef>;
}

impl IntoPyResult for () {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        Ok(vm.ctx.none())
    }
}

impl IntoPyResult for String {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        Ok(vm.ctx.new_str(self).into())
    }
}

pub fn pickle_object(vm: &VirtualMachine, obj: &PyObjectRef) -> PyResult<Vec<u8>> {
    let pickle = vm.import("pickle", 0)?;
    let dumped = vm.call_method(pickle.as_object(), "dumps", (obj.clone(),))?;
    let bytes = dumped
        .downcast_ref::<rustpython_vm::builtins::PyBytes>()
        .ok_or_else(|| vm.new_type_error("pickle.dumps must return bytes"))?;
    Ok(bytes.as_bytes().to_vec())
}

pub fn unpickle_object(vm: &VirtualMachine, data: &[u8]) -> PyResult<PyObjectRef> {
    let pickle = vm.import("pickle", 0)?;
    let py_bytes = vm.ctx.new_bytes(data.to_vec());
    vm.call_method(pickle.as_object(), "loads", (py_bytes,))
}

pub fn is_coroutine(vm: &VirtualMachine, obj: &PyObjectRef) -> PyResult<bool> {
    let asyncio = vm.import("asyncio", 0)?;
    vm.call_method(asyncio.as_object(), "iscoroutine", (obj.clone(),))?
        .try_into_value(vm)
}

pub fn run_on_event_loop(
    vm: &VirtualMachine,
    event_loop: &PyObjectRef,
    coro: PyObjectRef,
) -> PyResult<PyObjectRef> {
    await_coroutine_on_running_loop(vm, event_loop, coro)
}

pub fn await_coroutine_on_running_loop(
    vm: &VirtualMachine,
    event_loop: &PyObjectRef,
    coro: PyObjectRef,
) -> PyResult<PyObjectRef> {
    let asyncio = vm.import("asyncio", 0)?;
    let task = vm.call_method(asyncio.as_object(), "create_task", (coro,))?;
    loop {
        let is_done: bool = vm
            .call_method(task.as_object(), "done", ())?
            .try_into_value(vm)?;
        if is_done {
            return vm.call_method(task.as_object(), "result", ());
        }
        vm.call_method(event_loop.as_object(), "_run_once", ())?;
        drain_vm_queue(vm);
    }
}

pub fn read_contextvar(vm: &VirtualMachine, name: &str) -> Option<String> {
    let remote = vm.import("pulsing.core.remote", 0).ok()?;
    let name_key = vm.ctx.new_str(name);
    let cv = remote.get_attr(&name_key, vm).ok()?;
    let val = vm.call_method(cv.as_object(), "get", ()).ok()?;
    if vm.is_none(&val) {
        return None;
    }
    val.try_into_value::<PyStrRef>(vm)
        .ok()
        .map(|s| s.as_wtf8().to_string())
}

pub fn current_event_loop(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let asyncio = vm.import("asyncio", 0)?;
    let tasks = vm.import("asyncio.tasks", 0)?;
    if let Ok(current) = vm.call_method(tasks.as_object(), "current_task", ()) {
        if !vm.is_none(&current) {
            return vm.call_method(current.as_object(), "get_loop", ());
        }
    }
    if let Ok(running) = vm.call_method(asyncio.as_object(), "get_running_loop", ()) {
        if !vm.is_none(&running) {
            return Ok(running);
        }
    }
    vm.call_method(asyncio.as_object(), "get_event_loop", ())
}
