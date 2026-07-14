# pulsing-cli

Single-binary CLI for Pulsing (**Path B**).

## Layout

All **interactive UX** lives in this crate (`src/session/`). Downstream crates stay headless:

| Crate | Responsibility |
|-------|----------------|
| **pulsing-cli** | clap routing, immersive session, slash commands, plain render (future TUI/GUI) |
| **pulsing-forge** | agent loop, tools, LLM, sandbox |
| **pulsing-workspace** | init, journal, checkpoint, rollback |
| **pulsing-rpymod** | RustPython bindings for extension-mode workflows |

```
pulsing-cli/src/
  main.rs          # clap only → dispatch
  session/         # ★ interaction kernel
    mod.rs         # unified REPL loop
    commands.rs    # /help, /rollback, …
    mode.rs        # safe | workflow
    input.rs       # stdin (→ reedline later)
    render.rs      # plain text (→ ratatui / egui later)
    config.rs      # provider, workspace LLM config
    workspace.rs   # history, checkpoint, workflow list
  codex.rs         # thin entry → session::run_safe
  workflow.rs      # thin entry → session::run_workflow
  embed/           # RustPython extension mode
  workspace.rs     # pulsing init/history/… subcommands
```

## Modes

| Mode | Commands | Python |
|------|----------|--------|
| **Safe** (default) | `pulsing`, `pulsing "task"`, `pulsing agent` | None |
| **Extension** | `pulsing run`, `pulsing workflow` | RustPython + embedded sources |

Immersive session: `/help`, `/exit`, agent tasks, workflow rerun — all via `session/`.

## Build

```bash
just build-binary release=release
# target/release/pulsing  (~40MB release)
```

## Usage

```bash
pulsing init -g "Python CLI with pytest"
pulsing                    # session (safe)
pulsing run                # workflow → session
pulsing /help              # inside session
```

## Environment

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Live LLM providers |
| `PULSING_REPO_ROOT` | Repo root for extension-mode `sys.path` |
