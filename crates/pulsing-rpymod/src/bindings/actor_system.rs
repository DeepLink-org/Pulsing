use std::collections::HashMap;
use std::sync::Arc;

use pulsing_actor::actor::{ActorPath, NodeId};
use pulsing_actor::prelude::{ActorRef, ActorSystem};
use pulsing_actor::supervision::{BackoffStrategy, RestartPolicy, SupervisionSpec};
use pulsing_actor::system::ActorSystemCoreExt;
use rustpython_vm::builtins::PyUtf8StrRef;
use rustpython_vm::function::OptionalArg;
use rustpython_vm::{AsObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use super::message::{PyActorId, PyNodeId, PySystemConfig};
use super::python_actor::PythonActorWrapper;
use crate::interop::{current_event_loop, defer_async, IntoPyResult};
use crate::send_py::SendPy;

#[pyclass(module = "pulsing._core", name = "ActorSystem")]
#[derive(PyPayload)]
pub struct PyActorSystem {
    pub(crate) system: Arc<ActorSystem>,
}

impl std::fmt::Debug for PyActorSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyActorSystem")
            .field("node_id", &self.system.node_id().to_string())
            .finish()
    }
}

#[pyclass]
impl PyActorSystem {
    #[pystaticmethod]
    fn create(
        config: PyRef<PySystemConfig>,
        event_loop: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = config.inner.clone();
        defer_async(vm, event_loop, async move {
            let system = ActorSystem::new(inner).await.map_err(|e| e.to_string())?;
            Ok(system)
        })
    }

    #[pygetset]
    fn node_id(&self) -> PyNodeId {
        PyNodeId {
            inner: *self.system.node_id(),
        }
    }

    #[pygetset]
    fn addr(&self) -> String {
        self.system.addr().to_string()
    }

    #[pymethod]
    fn spawn(
        &self,
        actor: PyObjectRef,
        name: OptionalArg<PyUtf8StrRef>,
        _public: OptionalArg<bool>,
        restart_policy: OptionalArg<PyUtf8StrRef>,
        max_restarts: OptionalArg<u32>,
        min_backoff: OptionalArg<f64>,
        max_backoff: OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let system = Arc::clone(&self.system);
        let event_loop = current_event_loop(vm)?;
        let event_loop_for_async = SendPy::new(event_loop.clone());
        let policy_str = restart_policy
            .into_option()
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|| "never".to_string());
        let policy = match policy_str.to_lowercase().as_str() {
            "always" => RestartPolicy::Always,
            "on-failure" | "on_failure" => RestartPolicy::OnFailure,
            _ => RestartPolicy::Never,
        };
        let supervision = if matches!(policy, RestartPolicy::Never) {
            SupervisionSpec::never()
        } else {
            SupervisionSpec {
                policy,
                max_restarts: max_restarts.into_option().unwrap_or(3),
                backoff: BackoffStrategy::exponential(
                    std::time::Duration::from_secs_f64(min_backoff.into_option().unwrap_or(0.1)),
                    std::time::Duration::from_secs_f64(max_backoff.into_option().unwrap_or(30.0)),
                ),
                ..Default::default()
            }
        };
        let metadata = extract_metadata(vm, &actor)?;
        let name = name.into_option().map(|s| s.as_str().to_string());
        let actor = SendPy::new(actor);

        defer_async(vm, event_loop, async move {
            let actor = actor.into_inner();
            let event_loop = event_loop_for_async.into_inner();
            let actor_ref = match name {
                None => {
                    if !matches!(policy, RestartPolicy::Never) {
                        return Err(
                            "Anonymous actors do not support supervision/restart".to_string(),
                        );
                    }
                    let wrapper = PythonActorWrapper::new(actor, event_loop);
                    system
                        .spawning()
                        .metadata(metadata)
                        .spawn(wrapper)
                        .await
                        .map_err(|e| e.to_string())?
                }
                Some(name) => {
                    let name = if name.contains('/') {
                        name
                    } else {
                        format!("actors/{name}")
                    };
                    let path = if name.starts_with("system/") {
                        ActorPath::new_system(&name).map_err(|e| e.to_string())?
                    } else {
                        ActorPath::new(&name).map_err(|e| e.to_string())?
                    };
                    let wrapper = PythonActorWrapper::new(actor, event_loop);
                    system
                        .spawning()
                        .path(path)
                        .supervision(supervision)
                        .metadata(metadata)
                        .spawn(wrapper)
                        .await
                        .map_err(|e| e.to_string())?
                }
            };
            Ok(actor_ref)
        })
    }

    #[pymethod]
    fn shutdown(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let system = Arc::clone(&self.system);
        defer_async(vm, current_event_loop(vm)?, async move {
            system.shutdown().await.map_err(|e| e.to_string())?;
            Ok(())
        })
    }

    #[pymethod]
    fn refer(&self, actor_id: PyRef<PyActorId>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let system = Arc::clone(&self.system);
        let id = actor_id.inner;
        defer_async(vm, current_event_loop(vm)?, async move {
            let actor_ref = system.actor_ref(&id).await.map_err(|e| e.to_string())?;
            Ok(actor_ref)
        })
    }

    #[pymethod]
    fn actor_ref(&self, actor_id: PyRef<PyActorId>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.refer(actor_id, vm)
    }

    #[pymethod]
    fn members(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let system = Arc::clone(&self.system);
        defer_async(vm, current_event_loop(vm)?, async move {
            let members = system.members().await;
            Ok(members)
        })
    }

    #[pymethod]
    fn local_actor_names(&self, vm: &VirtualMachine) -> PyResult<Vec<PyObjectRef>> {
        Ok(self
            .system
            .local_actor_names()
            .into_iter()
            .map(|s| vm.ctx.new_str(s).into())
            .collect())
    }

    #[pymethod]
    fn resolve(
        &self,
        name: PyUtf8StrRef,
        node_id: OptionalArg<u128>,
        timeout: OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let system = Arc::clone(&self.system);
        let name = name.as_str().to_string();
        let node = node_id.into_option().map(NodeId::new);
        let timeout = timeout.into_option();
        defer_async(vm, current_event_loop(vm)?, async move {
            let name = if name.contains('/') {
                name
            } else {
                format!("actors/{name}")
            };
            let path = if name.starts_with("system/") {
                ActorPath::new_system(&name).map_err(|e| e.to_string())?
            } else {
                ActorPath::new(&name).map_err(|e| e.to_string())?
            };
            match timeout {
                None => system
                    .resolve_named(&path, node.as_ref())
                    .await
                    .map_err(|e| e.to_string()),
                Some(secs) => {
                    let deadline =
                        tokio::time::Instant::now() + std::time::Duration::from_secs_f64(secs);
                    let mut last_err = None;
                    while tokio::time::Instant::now() < deadline {
                        match system.resolve_named(&path, node.as_ref()).await {
                            Ok(r) => return Ok(r),
                            Err(e) => last_err = Some(e),
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    }
                    Err(last_err.unwrap().to_string())
                }
            }
        })
    }

    #[pymethod]
    fn resolve_named(
        &self,
        name: PyUtf8StrRef,
        node_id: OptionalArg<u128>,
        timeout: OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        self.resolve(name, node_id, timeout, vm)
    }
}

fn extract_metadata(vm: &VirtualMachine, actor: &PyObjectRef) -> PyResult<HashMap<String, String>> {
    let mut meta = HashMap::new();
    if let Ok(class) = actor.get_attr("__class__", vm) {
        if let Ok(name) = class.get_attr("__name__", vm) {
            if let Ok(s) = name.try_into_value::<PyUtf8StrRef>(vm) {
                meta.insert("python_class".to_string(), s.as_str().to_string());
            }
        }
        if let Ok(module) = class.get_attr("__module__", vm) {
            if let Ok(s) = module.try_into_value::<PyUtf8StrRef>(vm) {
                meta.insert("python_module".to_string(), s.as_str().to_string());
            }
        }
    }
    Ok(meta)
}

impl IntoPyResult for ActorRef {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        Ok(super::actor_ref::PyActorRef { inner: self }
            .into_ref(&vm.ctx)
            .into())
    }
}

impl IntoPyResult for Vec<pulsing_actor::cluster::MemberInfo> {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let items: Vec<PyObjectRef> = self
            .into_iter()
            .map(|m| {
                let dict = vm.ctx.new_dict();
                dict.set_item("node_id", vm.ctx.new_int(m.node_id.0).into(), vm)
                    .ok();
                dict.set_item("addr", vm.ctx.new_str(m.addr.to_string()).into(), vm)
                    .ok();
                dict.set_item("status", vm.ctx.new_str(format!("{:?}", m.status)).into(), vm)
                    .ok();
                dict.into()
            })
            .collect();
        Ok(vm.ctx.new_list(items).into())
    }
}

impl IntoPyResult for Arc<ActorSystem> {
    fn into_pyresult(self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        Ok(PyActorSystem { system: self }
            .into_ref(&vm.ctx)
            .into())
    }
}
