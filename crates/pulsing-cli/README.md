# pulsing-cli

Rust entry point for the Pulsing command-line interface (**Path B**: RustPython).

## Dual build modes

| Path | Command | Python runtime | Artifact |
|------|---------|----------------|----------|
| **A — wheel** | `just dev` / `just build-wheel` | User's CPython + `pulsing._core.so` (PyO3) | wheel / `pip install` |
| **B — binary** | `just build-binary` | **RustPython** (`rustpython_vm`) in-process | `target/release/pulsing` |

Path B does **not** link `libpython` or use PyO3 `embedded`. It is a pure-Rust Python interpreter.

## Build

```bash
just build-binary release=release
# target/release/pulsing
```

No `PYO3_PYTHON` required.

## Usage

```bash
./target/release/pulsing run examples/python/forge_agent_quickstart.py
./target/release/pulsing actor --help
```

## API parity (Path A ≡ Path B)

Both paths must expose the same ``pulsing._core`` Python API. Shared logic lives in
``pulsing-bindings-core``; contract tests in ``tests/python/test_core_api_surface.py``.

Path B is still catching up (``ActorSystem.create`` / ``spawn`` / policies / forge are stubs).
Do not add Path-specific branches in application Python when bindings can be fixed instead.

## Limitations (Path B today)

- **No CPython extensions**: `pulsing._core` (PyO3/maturin), `pydantic_core`, `hyperparameter` C extensions do not load.
- **Pure Python subset**: only packages compatible with RustPython run here.
- **Native bridge (planned)**: expose Actor/Forge to RustPython via `#[pymodule]` in Rust, calling `pulsing-actor` / `pulsing-forge` directly.

Path A (`maturin develop`) remains the full-fidelity development path until RustPython native modules land.

## Environment

| Variable | Purpose |
|----------|---------|
| `PULSING_REPO_ROOT` | Repo root; adds `python/` to `sys.path` |
