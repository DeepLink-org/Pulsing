//! Run Pulsing Python code inside a RustPython VM (Path B).

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use pulsing_rpymod::ensure_runtime;
use rustpython::InterpreterBuilder;
use rustpython::InterpreterBuilderExt;
use rustpython_vm as vm;
use rustpython_vm::compiler::Mode;
use rustpython_vm::import::import_builtin;

const BOOTSTRAP: &str = r#"
import asyncio
import inspect
import pulsing._core as _core
import pulsing.core

_orig_asyncio_run = asyncio.run

def _pulsing_asyncio_run(main, *args, **kwargs):
    async def _wrapper():
        async def _drainer():
            while True:
                _core._drain_vm_queue_sync()
                await asyncio.sleep(0)

        asyncio.create_task(_drainer())
        if inspect.iscoroutinefunction(main):
            return await main()
        if asyncio.iscoroutine(main):
            return await main
        return main

    return _orig_asyncio_run(_wrapper(), *args, **kwargs)

asyncio.run = _pulsing_asyncio_run

pulsing.core._cli_attach_from_native(_core.get_cli_actor_system())
"#;

fn with_vm<F>(f: F) -> Result<ExitCode>
where
    F: FnOnce(&vm::VirtualMachine) -> vm::PyResult<()>,
{
    ensure_runtime().context("failed to start ActorSystem for pulsing-cli")?;

    let builder = InterpreterBuilder::new().init_stdlib();
    let core_def = pulsing_rpymod::module_def(&builder.ctx);
    let interpreter = builder.add_native_module(core_def).interpreter();

    let code = interpreter.run(|vm| {
        setup_paths(vm)?;
        import_builtin(vm, "pulsing._core")?;
        run_source(vm, BOOTSTRAP)?;
        f(vm)
    });
    Ok(ExitCode::from(code as u8))
}

fn setup_paths(vm: &vm::VirtualMachine) -> vm::PyResult<()> {
    if let Some(root) = repo_root() {
        let python_src = root.join("python");
        if python_src.join("pulsing").is_dir() {
            let s = python_src.to_string_lossy();
            vm.insert_sys_path(vm.ctx.new_str(s.as_ref()).into())?;
        }
    }
    Ok(())
}

fn run_source(vm: &vm::VirtualMachine, source: &str) -> vm::PyResult<()> {
    let scope = vm.new_scope_with_builtins();
    let code = vm
        .compile(source, Mode::Exec, "<pulsing-cli>".to_owned())
        .map_err(|err| vm.new_syntax_error(&err, Some(source)))?;
    vm.run_code_obj(code, scope).map(|_| ())
}

/// Run ``pulsing.cli`` with the given arguments (post-binary argv slice).
pub fn delegate_to_python_cli(args: &[String]) -> Result<ExitCode> {
    let mut argv = vec![String::new()];
    argv.extend(args.iter().cloned());
    let source = format!(
        "import sys; sys.argv = {argv:?}; import runpy; \
         runpy.run_module('pulsing.cli', run_name='__main__', alter_sys=True)"
    );
    with_vm(|vm| run_source(vm, &source))
}

/// Execute a user ``.py`` script inside the RustPython VM.
pub fn run_python_script(script: &Path, script_args: &[String]) -> Result<ExitCode> {
    let script = script
        .canonicalize()
        .with_context(|| format!("script not found: {}", script.display()))?;

    let mut argv = vec![script.to_string_lossy().into_owned()];
    argv.extend(script_args.iter().cloned());
    let path = script.to_string_lossy();
    let source = format!(
        "import sys; sys.argv = {argv:?}; import runpy; \
         runpy.run_path({path:?}, run_name='__main__')"
    );
    with_vm(|vm| run_source(vm, &source))
}

fn repo_root() -> Option<PathBuf> {
    if let Ok(root) = env::var("PULSING_REPO_ROOT") {
        let p = PathBuf::from(root);
        if p.join("python/pulsing").is_dir() {
            return Some(p);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        for dir in cwd.ancestors() {
            if dir.join("python/pulsing").is_dir() {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rustpython_hello() {
        let interpreter = InterpreterBuilder::new().init_stdlib().interpreter();
        interpreter.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            let source = r#"print("ok")"#;
            let code = vm
                .compile(source, Mode::Exec, "<test>".to_owned())
                .expect("compile");
            vm.run_code_obj(code, scope).expect("run");
        });
    }
}
