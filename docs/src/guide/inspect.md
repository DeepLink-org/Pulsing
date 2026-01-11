# Inspect (CLI)

`pulsing inspect` joins a cluster (via seeds) and prints a human-friendly snapshot:

- cluster members (node id / addr / status)
- named actor distribution (best-effort)

## Usage

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

## Notes

- `--seeds` is required.
- If your seeds are local (`127.0.0.1` / `localhost`), the CLI binds to `127.0.0.1:0` for connectivity.

