# Forge Session REPL

Interactive REPL for driving Forge tools without an LLM — trace record/replay for debugging and regression.

---

## Entry points

```bash
pulsing forge repl --cwd .
python -m pulsing.forge.repl   # Python-only
cargo run -p pulsing-forge --bin pulsing-forge-repl -- --cwd .   # dev: Rust only
```

Examples: `Read --file_path README.md` · `/tools` · `/approve ask`

---

## Purpose

- JSONL trace save/replay (`--trace`, `--record`, `--replay-all --verify`)
- Complements pytest for integration and regression tests

Full spec (Rust vs Python, slash commands): **`session-repl.zh.md`** (Chinese, primary).
