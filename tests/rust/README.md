# Rust Tests

Rust tests are located in the respective crates:

```bash
# Run pulsing-actor unit tests
cargo test -p pulsing-actor

# Run all tests
cargo test --workspace --exclude pulsing-py
```

## Test Locations

- `crates/pulsing-actor/src/**/*.rs` - Unit tests (inline `#[cfg(test)]` modules)
- `crates/pulsing-actor/tests/` - Integration tests
- `crates/pulsing-workspace/src/` - Workspace journal tests
