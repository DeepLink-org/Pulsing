use pulsing_rpymod::{module_def, PyActorSystem, PyNodeId, PySystemConfig};
use rustpython::InterpreterBuilder;
use rustpython::InterpreterBuilderExt;
use rustpython_vm::class::PyClassImpl;
use rustpython_vm::AsObject;

fn has_attr(
    vm: &rustpython_vm::VirtualMachine,
    obj: rustpython_vm::PyObjectRef,
    name: &str,
) -> bool {
    let key = vm.ctx.intern_str(name);
    obj.get_attr(key, vm).is_ok()
}

fn type_attr_names(
    vm: &rustpython_vm::VirtualMachine,
    type_obj: rustpython_vm::PyObjectRef,
) -> Vec<String> {
    type_obj
        .dir(vm)
        .map(|list| {
            list.borrow_vec()
                .iter()
                .filter_map(|item| {
                    item.downcast_ref::<rustpython_vm::builtins::PyStr>()
                        .map(|s| s.as_wtf8().to_string())
                        .filter(|name| !name.starts_with('_'))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn runtime_type_exposes_spawn() {
    let builder = InterpreterBuilder::new().init_stdlib();
    let core_def = module_def(&builder.ctx);
    let interpreter = builder.add_native_module(core_def).interpreter();
    interpreter.enter(|vm| {
        let typ = PyActorSystem::make_static_type();
        let obj = typ.as_object().to_owned();
        let names = type_attr_names(vm, obj.clone());
        eprintln!("ActorSystem runtime attrs: {:?}", names);
        eprintln!(
            "ActorSystem METHOD_DEFS: {:?}",
            PyActorSystem::METHOD_DEFS
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
        );
        assert!(
            has_attr(vm, obj, "spawn"),
            "spawn missing on type, attrs={names:?}"
        );
    });
}

#[test]
fn runtime_nodeid_exposes_test_return_none_issue() {
    let builder = InterpreterBuilder::new().init_stdlib();
    let core_def = module_def(&builder.ctx);
    let interpreter = builder.add_native_module(core_def).interpreter();
    interpreter.enter(|vm| {
        let typ = PyNodeId::make_static_type();
        let obj = typ.as_object().to_owned();
        let names = type_attr_names(vm, obj.clone());
        eprintln!("NodeId runtime attrs: {:?}", names);
        eprintln!(
            "has test_pyobj attr: {}",
            has_attr(vm, obj.clone(), "test_pyobj")
        );
        assert!(has_attr(vm, obj, "generate"));
    });
}

#[test]
fn runtime_system_config_pyobject_return_method() {
    let builder = InterpreterBuilder::new().init_stdlib();
    let core_def = module_def(&builder.ctx);
    let interpreter = builder.add_native_module(core_def).interpreter();
    interpreter.enter(|vm| {
        let typ = PySystemConfig::make_static_type();
        let obj = typ.as_object().to_owned();
        eprintln!(
            "SystemConfig has test_return_none: {}",
            has_attr(vm, obj.clone(), "test_return_none")
        );
        eprintln!(
            "SystemConfig has with_addr: {}",
            has_attr(vm, obj, "with_addr")
        );
    });
}

#[test]
fn import_builtin_exposes_spawn() {
    let builder = InterpreterBuilder::new().init_stdlib();
    let core_def = module_def(&builder.ctx);
    let interpreter = builder.add_native_module(core_def).interpreter();
    interpreter.enter(|vm| {
        rustpython_vm::import::import_builtin(vm, "pulsing._core").expect("import_builtin");
        let modules = vm.sys_module.get_attr("modules", vm).expect("modules");
        let module = modules
            .get_item("pulsing._core", vm)
            .expect("pulsing._core in sys.modules");
        let cls = module.get_attr("ActorSystem", vm).expect("ActorSystem");
        assert!(
            has_attr(vm, cls, "spawn"),
            "spawn missing after import_builtin"
        );
    });
}
