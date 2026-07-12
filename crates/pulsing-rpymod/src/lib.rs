//! RustPython native module ``pulsing._core``.

#[macro_use]
extern crate rustpython_derive;

mod bindings;
mod interop;
mod runtime;
mod send_py;

pub use bindings::PyNodeId;
pub use bindings::{PyActorSystem, PySystemConfig};
pub use interop::{drain_vm_queue, schedule_vm_work};
pub use runtime::{ensure_runtime, start_vm_drainer};

use rustpython_vm::builtins::PyModuleDef;
use rustpython_vm::Context;

pub fn module_def(ctx: &Context) -> &'static PyModuleDef {
    pulsing_core::module_def(ctx)
}

#[pymodule(name = "pulsing._core")]
mod pulsing_core {
    use rustpython_vm::builtins::{PyModule, PyUtf8StrRef};
    use rustpython_vm::class::PyClassImpl;
    use rustpython_vm::extend_module;
    use rustpython_vm::function::OptionalArg;
    use rustpython_vm::{AsObject, Py, PyPayload, PyRef, PyResult, VirtualMachine};

    use crate::bindings::{
        PyActorId, PyActorRef, PyActorSystem, PyMessage, PyNodeId, PyStreamMessage, PyStreamReader,
        PyStreamWriter, PySystemConfig, PyZeroCopyDescriptor,
    };
    use crate::interop::drain_vm_queue;

    pub(crate) fn module_exec(vm: &VirtualMachine, module: &Py<PyModule>) -> PyResult<()> {
        __module_exec(vm, module);
        extend_module!(vm, module, {
            "NodeId" => PyNodeId::make_static_type(),
            "ActorId" => PyActorId::make_static_type(),
            "Message" => PyMessage::make_static_type(),
            "ZeroCopyDescriptor" => PyZeroCopyDescriptor::make_static_type(),
            "StreamReader" => PyStreamReader::make_static_type(),
            "StreamWriter" => PyStreamWriter::make_static_type(),
            "StreamMessage" => PyStreamMessage::make_static_type(),
            "SystemConfig" => PySystemConfig::make_static_type(),
            "ActorRef" => PyActorRef::make_static_type(),
            "ActorSystem" => PyActorSystem::make_static_type(),
        });
        Ok(())
    }

    #[pyfunction]
    fn get_cli_actor_system(vm: &VirtualMachine) -> PyResult<PyRef<PyActorSystem>> {
        let system =
            crate::runtime::ensure_runtime().map_err(|e| vm.new_runtime_error(e.to_string()))?;
        Ok(PyActorSystem { system }.into_ref(&vm.ctx))
    }

    #[pyfunction]
    fn _drain_vm_queue_sync(vm: &VirtualMachine) -> PyResult<()> {
        drain_vm_queue(vm);
        Ok(())
    }

    #[pyfunction]
    fn init_distributed_tracing(
        _service_name: OptionalArg<PyUtf8StrRef>,
        _console_output: OptionalArg<bool>,
    ) -> PyResult<()> {
        Ok(())
    }

    #[pyfunction]
    fn shutdown_distributed_tracing() -> PyResult<()> {
        Ok(())
    }

    #[pyattr(name = "__version__")]
    const VERSION: &str = env!("CARGO_PKG_VERSION");
}
