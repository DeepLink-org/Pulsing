# pulsing-cli

RustPython-based actor CLI and workspace tools.

```bash
just build-binary
pulsing init --template minimal
pulsing run                  # execute .pulsing/workflows/example.py once
pulsing run my_workflow.py
pulsing history
pulsing checkpoint -m "working state"
pulsing rollback
pulsing actor my_app.Counter --addr 127.0.0.1:8000
pulsing inspect actors --endpoint 127.0.0.1:8000
```

Workspace commands run in Rust. Workflow scripts and actor/inspect commands use
the embedded RustPython VM and the Python sources. Set `PULSING_REPO_ROOT` when
running outside the checkout.

There is no default agent session, tool REPL, LLM provider configuration, or GUI.
Running without a subcommand prints help.

Build with the checked-in `Cargo.lock` (`cargo build --locked -p pulsing-cli`).
RustPython 0.5 needs the pinned libc and the shared malachite-bigint 0.9 dependency
used by its VM and pymath; unconstrained re-resolution can select incompatible
upstream versions.
