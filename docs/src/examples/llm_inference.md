# LLM Inference (overview)

Pulsing is a **general-purpose distributed actor framework** and also a good fit for **LLM inference services**, especially when you need:

- a router + worker architecture
- distributed scheduling / load awareness
- streaming responses (`ask_stream`)

This page is currently an overview (Draft). See:

- `docs/src/design/http2-transport.md` for the HTTP/2 streaming protocol design
- `docs/src/design/load_sync.md` for load sync concepts

## Suggested architecture

- **Router**: accepts client requests, chooses a worker, forwards request
- **Workers**: host model replicas, expose `generate` / `generate_stream`

## Next step

If you want this page to become a runnable example, tell me which backend you want:

- `transformers` + `torch`
- `vllm`
- `triton` / custom engine
