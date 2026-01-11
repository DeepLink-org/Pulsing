# Actor List (CLI)

`pulsing actor list` is a lightweight **observer** tool that queries actor lists via HTTP endpoints.

## When to use

- Verify whether a node is reachable
- Check which actors exist on a node
- Get a quick view of a cluster without joining the gossip cluster

## Single node

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### Show internal/system actors

```bash
pulsing actor list --endpoint 127.0.0.1:8000 --all_actors True
```

### JSON output

```bash
pulsing actor list --endpoint 127.0.0.1:8000 --json True
```

## Cluster (via seeds)

```bash
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

## Notes

- This command uses HTTP/2 (h2c) requests and does **not** require joining the cluster.
- If a node does not expose the HTTP endpoints, it will appear as unreachable.

