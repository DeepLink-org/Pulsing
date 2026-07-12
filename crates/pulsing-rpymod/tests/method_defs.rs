use pulsing_rpymod::{PyActorSystem, PyNodeId, PySystemConfig};
use rustpython_vm::class::PyClassImpl;

#[test]
fn actor_system_method_defs_include_spawn() {
    let names: Vec<&str> = PyActorSystem::METHOD_DEFS.iter().map(|m| m.name).collect();
    eprintln!("ActorSystem METHOD_DEFS: {:?}", names);
    assert!(
        names.iter().any(|n| *n == "spawn"),
        "missing spawn in {:?}",
        names
    );
}

#[test]
fn compare_method_defs_counts() {
    eprintln!(
        "NodeId METHOD_DEFS: {:?}",
        PyNodeId::METHOD_DEFS
            .iter()
            .map(|m| m.name)
            .collect::<Vec<_>>()
    );
    eprintln!(
        "ActorSystem METHOD_DEFS: {:?}",
        PyActorSystem::METHOD_DEFS
            .iter()
            .map(|m| m.name)
            .collect::<Vec<_>>()
    );
    assert!(PyActorSystem::METHOD_DEFS.len() > 2);
}
