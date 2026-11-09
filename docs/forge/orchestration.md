# Pulsing Forge — MCP-native multi-agent orchestration tier

Pulsing Forge adds an **agent-team orchestration layer** on top of the Pulsing
actor philosophy: every agent gets a **mailbox**, every letter lands in a
**journaled ledger**, and coordination is an `Orchestrator` trait instead of a
hardcoded loop. It is MCP-native — the entire agent-side contract is four MCP
tools (`pulsing_send`, `pulsing_broadcast`, `pulsing_drain_mail`,
`pulsing_health`), so any MCP-capable agent runtime (codex, claude-code,
cursor, or a plain script) can join a team with zero runtime modifications.

## Layering with the rest of Pulsing

| Layer | Crate | Responsibility |
|---|---|---|
| Actor runtime | `pulsing-actor` | distributed actors, transport, supervision |
| **Orchestration tier** | `pulsing-forge-core` | gates, closure semantics, orchestrator trait + 6 built-ins |
| **Live agent tier** | `pulsing-forge-hub` | TideHub mailbox + journal + MCP control port + shop scheduler |
| **CLI** | `pulsing-forge-runner` | `forge run <scenario.toml>`, `forge attach`, dashboard |

Forge deliberately keeps its mailbox in-process (single binary, file-backed
journal). A natural follow-up is rebasing TideHub onto `pulsing-actor`
transports for multi-node mailboxes; the trait boundaries were drawn to allow
exactly that.

## The five-minute path

```bash
cargo build --release -p pulsing-forge-runner
./target/release/forge run examples/forge/leak-review.toml
```

A deterministic write-review loop runs end-to-end (writer drafts, reviewers
gate, judge arbitrates, dual-channel closure). Point `adapter = "codex"` in a
scenario to swap the deterministic tier for live LLM agents; see
`docs/forge/adapters.md`.

## What is in this directory

- `failure-taxonomy.md` — 16 failure modes of agent teams and the structural
  fix shipped for each (the design driver of the whole crate set)
- `scenario-schema.md` / `ledger-format.md` — the two file formats
- `adapters.md` — deterministic / sleep-executor / codex (live LLM) adapters
- `design-notes.md` — why no LLM router, declarative topology, forced closure
