//! CLI help text — safe mode vs extension mode.

pub const ABOUT: &str = "Pulsing — AI application server (single binary)";

pub const LONG_ABOUT: &str = "\
Pulsing ships as one binary with two modes:

  SAFE MODE (default)
    Rust agent + Forge tools + workspace journal. No user Python scripts.
      pulsing                     interactive agent
      pulsing \"fix the tests\"     one-shot task
      pulsing init -g \"…\"         bootstrap workspace (optional LLM guide)

  EXTENSION MODE (opt-in)
    Immersive workflow session — stays in the CLI after success.
      pulsing run                     default workflow (example.py)
      pulsing run my_workflow.py      explicit script
    On failure, returns to the shell with recovery hints.

  Workspace checkpoints (fault tolerance):
      pulsing history | checkpoint | rollback

  Desktop GUI (egui):
      pulsing gui                 Desktop chat window (egui)

Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live models; otherwise demo LLM is used.";

pub const AFTER_HELP: &str = "\
Examples:
  pulsing init -g \"Python CLI with pytest\"
  pulsing \"add error handling to src/main.rs\"
  pulsing gui
  pulsing run
  pulsing rollback

Workflow failed? You return to the shell — then:
  pulsing rollback && pulsing \"fix .pulsing/workflows/example.py\"";

#[allow(dead_code)]
pub const EXTENSION_MODE_HINT: &str = "\
note: `pulsing run` opens an immersive workflow session (Codex-like). \
Safe mode: `pulsing` or `pulsing \"task\"`.";

pub const EXTENSION_UNAVAILABLE: &str = "\
extension mode is not available in this build (Python sources not embedded).

Use safe mode instead — no Python required:
  pulsing                  # interactive agent
  pulsing \"your task\"      # one-shot

Developers: set PULSING_REPO_ROOT or run from the Pulsing repository.";

pub const LEGACY_HINT: &str = "\
note: legacy Python CLI (actor, inspect, …) — extension mode. \
Prefer `pulsing` for the Rust agent.";
