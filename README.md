# Pulsing Overview

Pulsing is a load- and KV-cache-aware LLM inference service system. It focuses on multi-tenant and high-concurrency scenarios: by dynamically sensing request cost, memory layout, and cache hit rate, Pulsing improves overall throughput and reduces tail latency, while keeping pluggable support for popular inference backends such as vLLM and SGLang.

This repository is an independently maintained fork from the `ai-dynamo/dynamo` project (current baseline version: v0.7.0; a precise upstream commit can later be recorded in the form `upstream: ai-dynamo/dynamo@<commit>`). Pulsing inherits Dynamo’s decoupled inference architecture and Rust+Python co-design, and evolves the system with targeted improvements in routing/scheduling strategies and structural maintainability.

Acknowledgements: we thank all contributors of the `ai-dynamo/dynamo` project for their high-quality open-source work. This project continues to follow the Apache-2.0 license and preserves the original copyright
and attribution statements in all derivative distributions.

## Build project
```shell
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# build wheel
cd lib/bindings/python
uv build --wheel --python 3.12
```

## ABI-friendly binary build

ABI-friendly (manylinux-compatible) wheels are recommended so that
prebuilt binaries can run reliably across different Linux distributions
without requiring users to rebuild from source.

### 1. Environment setup

```shell
pip install maturin  # build tool
pip install ziglang  # used for ABI-friendly linking
```

### 2. Build wheel package

Run the following in the `lib/bindings/python` directory:

```shell
cd lib/bindings/python
maturin pep517 build-wheel \
	--auditwheel repair --manylinux \
	--zig --compatibility manylinux_2_24
```