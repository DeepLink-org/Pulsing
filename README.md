# Pulsing Overview

Pulsing is a load- and KV-cache-aware LLM inference service system. It focuses on multi-tenant and high-concurrency scenarios: by dynamically sensing request cost, memory layout, and cache hit rate, Pulsing improves overall throughput and reduces tail latency, while keeping pluggable support for popular inference backends such as vLLM and SGLang.

This repository is an independently maintained fork from the `ai-dynamo/dynamo` project (current baseline version: v0.7.0; a precise upstream commit can later be recorded in the form `upstream: ai-dynamo/dynamo@<commit>`). Pulsing inherits Dynamo’s decoupled inference architecture and Rust+Python co-design, and evolves the system with targeted improvements in routing/scheduling strategies and structural maintainability.

Acknowledgements: we thank all contributors of the `ai-dynamo/dynamo` project for their high-quality open-source work. This project continues to follow the Apache-2.0 license and preserves the original copyright
and attribution statements in all derivative distributions.