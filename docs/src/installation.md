# Installation

This guide covers how to install Pulsing and get started with development.

## Prerequisites

Before installing Pulsing, ensure you have:

- **Python 3.10+** - Pulsing requires Python 3.10 or later
- **Rust toolchain** - For building the native extension
- **Linux/macOS** - Currently supported platforms

## Installation Methods

### Method 1: From Source (Development)

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/reiase/pulsing.git
cd pulsing

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin for building Python extensions
pip install maturin

# Build and install in development mode
maturin develop
```

### Method 2: From PyPI (Coming Soon)

```bash
pip install pulsing
```

## Verify Installation

After installation, verify everything works:

```python
import asyncio
from pulsing.actor import as_actor, create_actor_system, SystemConfig

@as_actor
class HelloActor:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

async def main():
    # Create actor system
    system = await create_actor_system(SystemConfig.standalone())

    # Create actor
    hello = await HelloActor.local(system)

    # Call method
    result = await hello.greet("World")
    print(result)  # Hello, World!

    # Cleanup
    await system.shutdown()

asyncio.run(main())
```

If you see "Hello, World!" printed, the installation was successful!

## Development Setup

For contributing to Pulsing:

### 1. Clone and Setup

```bash
git clone https://github.com/reiase/pulsing.git
cd pulsing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install development dependencies
pip install maturin pytest pytest-asyncio
```

### 2. Build

```bash
# Development build (with debug symbols)
maturin develop

# Release build (optimized)
maturin develop --release
```

### 3. Run Tests

```bash
# Run Python tests
pytest tests/

# Run Rust tests
cargo test --workspace
```

## Platform-Specific Notes

### Linux

No special requirements. All features are fully supported.

### macOS

- Apple Silicon (M1/M2): Fully supported
- Intel Macs: Fully supported

### Windows

Windows support is experimental. For best results, use WSL2 with Ubuntu.

## Troubleshooting

### Build Errors

**Problem:** `cargo build` fails with missing dependencies

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
```

**Problem:** `maturin develop` fails

**Solution:**

```bash
# Ensure you have the latest maturin
pip install --upgrade maturin

# Try with verbose output
maturin develop -v
```

### Runtime Errors

**Problem:** Import error after installation

**Solution:**

```bash
# Rebuild the extension
maturin develop --release
```

**Problem:** Port already in use

**Solution:**

```python
# Use a different port
config = SystemConfig.with_addr("0.0.0.0:8001")  # Change port
```

## Next Steps

Now that you have Pulsing installed:

1. **Quick Start** - Follow the [Quick Start Guide](quickstart/index.md)
2. **Actors Guide** - Learn about [Actors](guide/actors.md)
3. **Examples** - Explore [Example Code](examples/index.md)
4. **API Reference** - Check the [API Documentation](api_reference.md)
