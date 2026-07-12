//! ``pulsing-forge-repl`` — Rust Forge session REPL binary for ``pulsing forge repl``.

fn main() -> anyhow::Result<()> {
    pulsing_forge::cli::run_repl_from_iter(std::env::args())
}
