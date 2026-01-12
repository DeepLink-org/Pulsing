# Security Guide

Guide to securing your Pulsing cluster with TLS encryption.

## Overview

Pulsing supports **passphrase-based mTLS (Mutual TLS)** for secure cluster communication. This innovative design provides:

- **Zero-configuration PKI**: No need to generate or distribute certificates manually
- **Passphrase-based access**: Nodes with the same passphrase can join the cluster
- **Cluster isolation**: Different passphrases create completely isolated clusters
- **Mutual authentication**: Both server and client verify each other's certificates

!!! danger "Pickle serialization risk (RCE)"
    Pulsing currently uses **Python Pickle** for Python-to-Python message payloads.
    **Never accept untrusted payloads** or expose Pulsing ports to an untrusted network.

    Production guidance:

    - **Enable mTLS** by setting a `passphrase` (required for any real deployment)
    - **Network isolation** (private VPC/subnet + firewall) is still recommended
    - Treat the cluster as a **trusted boundary** until non-pickle codecs are the default

## Enabling TLS

### Development Mode (No TLS)

By default, Pulsing uses cleartext HTTP/2 (h2c) for easy debugging:

```python
from pulsing.actor import SystemConfig, create_actor_system

# No passphrase - uses cleartext HTTP/2
config = SystemConfig.with_addr("0.0.0.0:8000")
system = await create_actor_system(config)
```

### Production Mode (mTLS)

To enable TLS encryption, simply set a passphrase:

```python
# Set passphrase - automatically enables mTLS
config = SystemConfig.with_addr("0.0.0.0:8000").with_passphrase("my-cluster-secret")
system = await create_actor_system(config)
```

## Multi-Node Cluster with TLS

All nodes in a cluster must use the **same passphrase** to communicate:

```python
# Node 1: Seed node with TLS
config1 = SystemConfig.with_addr("0.0.0.0:8000").with_passphrase("shared-secret")
system1 = await create_actor_system(config1)

# Node 2: Join cluster with same passphrase
config2 = (
    SystemConfig.with_addr("0.0.0.0:8001")
    .with_seeds(["192.168.1.1:8000"])
    .with_passphrase("shared-secret")  # Must match!
)
system2 = await create_actor_system(config2)
```

!!! warning "Passphrase Mismatch"
    Nodes with different passphrases cannot communicate. The TLS handshake will fail.

## Cluster Isolation

Different passphrases create completely isolated clusters:

```python
# Cluster A
cluster_a = SystemConfig.with_addr("0.0.0.0:8000").with_passphrase("secret-a")

# Cluster B (different passphrase)
cluster_b = SystemConfig.with_addr("0.0.0.0:9000").with_passphrase("secret-b")

# cluster_a and cluster_b cannot communicate
```

## How It Works

Pulsing uses a **deterministic CA derivation** approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    Passphrase (口令)                        │
│                         "my-secret"                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HKDF-SHA256
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Deterministic CA Certificate                   │
│         (Same passphrase → Same CA cert/key)                │
│         Algorithm: Ed25519 | Validity: 10 years             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Signs
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Node Certificate (per node)                    │
│         (Each node generates its own, signed by CA)         │
│         CN: "Pulsing Node <uuid>" | SAN: pulsing.internal   │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Mutual Authentication** | Both server and client present certificates |
| **Passphrase = Access** | Only nodes knowing the passphrase can join |
| **Zero-config PKI** | No manual certificate generation/distribution |
| **Deterministic CA** | Same passphrase → Same CA (all nodes trust it) |
| **Isolated Clusters** | Different passphrases = completely separate |

## Security Best Practices

### 1. Use Strong Passphrases

!!! tip "Passphrase Strength"
    Use high-entropy random strings for production:
    ```python
    # Good: High entropy
    passphrase = "aX9#mK2$nL5@pQ8&"

    # Bad: Weak/predictable
    passphrase = "password123"
    ```

### 2. Environment Variables

Store passphrases in environment variables, not code:

```python
import os

passphrase = os.environ.get("PULSING_PASSPHRASE")
if passphrase:
    config = SystemConfig.with_addr("0.0.0.0:8000").with_passphrase(passphrase)
else:
    # Development mode - no TLS
    config = SystemConfig.with_addr("0.0.0.0:8000")
```

### 3. Rotate Passphrases

To rotate passphrases:

1. Deploy new nodes with the new passphrase
2. Gradually migrate actors to new nodes
3. Decommission old nodes

!!! note "Rolling Updates"
    Nodes with different passphrases cannot communicate. Plan for a brief transition period.

### 4. Network Segmentation

Even with TLS, use network-level security:

- Deploy in private VPCs/subnets
- Use firewalls to restrict access
- Consider VPN for cross-datacenter communication

## Comparison with Other Frameworks

| Aspect | Pulsing | Ray | Traditional mTLS |
|--------|---------|-----|------------------|
| **Configuration** | 1 line of code | Multiple config files | PKI infrastructure required |
| **Certificate Management** | None needed | Need to distribute certs | Need CA + cert rotation |
| **New Node Join** | Know passphrase | Pre-configure certificates | Issue new certificates |
| **Cluster Isolation** | Different passphrase | Different cert system | Different CA |
| **Crypto Algorithm** | Ed25519 + mTLS | TLS 1.2/1.3 | Depends on config |

## Limitations

!!! warning "Current Limitations"
    - **No authorization**: Any actor can call any actor (authentication only, not authorization)
    - **Pickle serialization**: Message payloads still use Pickle (plan to replace with msgpack)
    - **No cert rotation**: Changing passphrase requires cluster restart

## Example: Secure Distributed Counter

```python
import os
from pulsing.actor import init, shutdown, remote

# Get passphrase from environment
PASSPHRASE = os.environ.get("PULSING_SECRET", None)

@remote
class SecureCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def get(self) -> int:
        return self.value

    def increment(self, n: int = 1) -> int:
        self.value += n
        return self.value

async def main():
    # Create config with optional TLS
    config = SystemConfig.with_addr("0.0.0.0:8000")
    if PASSPHRASE:
        config = config.with_passphrase(PASSPHRASE)

    system = await create_actor_system(config)

    # Spawn secure counter
    counter = await SecureCounter.local(system, init_value=0)
    await system.spawn(counter, "secure-counter", public=True)

    print("Secure counter running...")
    print(f"TLS enabled: {PASSPHRASE is not None}")
```

## Next Steps

- Learn about [Remote Actors](remote_actors.md) for cluster communication
- Check [HTTP2 Transport](../design/http2-transport.md) for transport details
- Read [Semantics](semantics.md) for message delivery guarantees
