//! Send wrapper for Python object references used across tokio tasks.
//!
//! SAFETY: RustPython objects are only accessed on the VM thread via `schedule_vm_work`.

pub struct SendPy(pub rustpython_vm::PyObjectRef);

unsafe impl Send for SendPy {}
unsafe impl Sync for SendPy {}

impl SendPy {
    pub fn new(obj: rustpython_vm::PyObjectRef) -> Self {
        Self(obj)
    }

    pub fn into_inner(self) -> rustpython_vm::PyObjectRef {
        self.0
    }
}
