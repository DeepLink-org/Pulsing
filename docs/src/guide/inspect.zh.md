# 巡检（CLI）

`pulsing inspect` 会通过 seeds 加入集群，并输出一个便于排障的快照：

- 集群成员（node id / addr / status）
- 命名 actors 的分布（best-effort）

## 用法

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

## 说明

- `--seeds` 为必填参数。
- 如果 seeds 是本地地址（`127.0.0.1` / `localhost`），CLI 会绑定 `127.0.0.1:0` 以保证连通性。

