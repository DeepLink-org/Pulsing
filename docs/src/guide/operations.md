# CLI Operations

## Start an actor

Provide an importable user class implementing the actor API:

```bash
pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10
```

Options before `--` configure the node and actor registration.
Options after `--` are passed to the actor constructor.
Use `--seeds 127.0.0.1:8000` to join an existing cluster.

## Inspect a cluster

Inspection uses HTTP observer mode and does not join the cluster.

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
pulsing inspect actors --endpoint 127.0.0.1:8000 --detailed
pulsing inspect actors --seeds 127.0.0.1:8000 --json
pulsing inspect metrics --seeds 127.0.0.1:8000
pulsing inspect watch --seeds 127.0.0.1:8000 --interval 2
```

Use `--timeout` to bound requests and `--best_effort` to continue when a node
is unavailable. `pulsing actor list` has been replaced by `pulsing inspect actors`.

## Workspace files

```bash
pulsing init --template minimal
pulsing checkpoint -m "working state"
pulsing history
pulsing rollback
```

The Rust CLI can execute a Python workflow once with `pulsing run [SCRIPT]`.
There are no built-in inference servers, benchmark commands, or agent sessions.
