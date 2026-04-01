# Python API Reference

This page contains the complete auto-generated API documentation for Pulsing's Python interface.

## Installation

Pulsing requires Python 3.10+ and can be installed via pip:

```bash
pip install pulsing
```

For development, clone the repository and install in development mode:

```bash
git clone https://github.com/DeepLink-org/pulsing
cd pulsing
pip install -e .
```

## Core Module

::: pulsing

## Actor Module

::: pulsing.core

## Agent Module

::: pulsing.agent

## Queue Module

::: pulsing.streaming

## Subprocess Module

`pulsing.subprocess` provides a subprocess-compatible synchronous API.
Without `resources`, calls fall back to Python's native `subprocess`.
When `resources` is provided and `USE_POLSING_SUBPROCESS=1` is set, the
command is executed through Pulsing's backend.

::: pulsing.subprocess
