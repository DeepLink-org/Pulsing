# Bench (CLI)

`pulsing bench` runs load tests against an inference endpoint (typically an OpenAI-compatible router).

## Basic usage

```bash
pulsing bench gpt2 --url http://localhost:8080
```

## Notes

- The benchmark extension is optional. If you see `pulsing._bench module not found`, install it with:

```bash
maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
```

