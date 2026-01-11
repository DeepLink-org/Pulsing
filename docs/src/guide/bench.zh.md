# 压测（CLI）

`pulsing bench` 用于对推理端点进行压测（通常是 OpenAI 兼容 Router）。

## 基本用法

```bash
pulsing bench gpt2 --url http://localhost:8080
```

## 说明

- Benchmark 扩展是可选的。如果出现 `pulsing._bench module not found`，需要单独安装：

```bash
maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
```

