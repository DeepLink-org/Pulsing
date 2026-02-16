# 按名字解析 Actor、不依赖具体类型（Name-Only Resolve）

**已实现**：`resolve(name)` 返回带 `.as_any()` 的包装；`ref.as_any()` 或 `as_any(ref)` 得到可转发任意方法调用的 proxy。

## 问题

当前要「按名字拿到可调用的 actor」有两种方式：

1. **类型化解析**：`await SomeClass.resolve("channel.discord")`
   - 得到带类型的 `ActorProxy`，可 `await proxy.send_text(...)`
   - 调用方必须知道并 import 具体类（如 `DiscordChannel`）

2. **底层解析**：`await pul.resolve("channel.discord")`
   - 得到 `ActorRef`，只能 `ref.ask(msg)`，不能 `proxy.method(...)`
   - 调用方要自己拼装 `__call__` / 协议格式

像 nanobot 这种「按 channel 名字发消息」的场景，调用方只知道名字（如 `"discord"`），不想依赖具体 channel 类型，因此只能自己维护 `name -> 类` 的映射再做 `XxxChannel.resolve(name)`。希望框架能提供「按名字解析 + 直接调方法」的能力。

---

## 现有实现要点

- `pul.resolve(name)` 已存在，返回 `ActorRef`。
- `ActorProxy(actor_ref, method_names, async_methods)`：
  - `method_names is None` 时，`__getattr__` 不校验名字，任意属性都会返回 `_MethodCaller`（即**已支持「任意方法名」**）。
  - 调用通过现有协议发到 actor（`__call__` / method + args/kwargs），actor 端用 `getattr(instance, method)` 分发，无需改动。
- 区别只在于：**谁构造 Proxy**（带不带类型信息）、以及**是否知道哪些方法是 async**（影响是否走 streaming 路径）。

因此「按名字解析并支持动态方法调用」不需要改协议或 actor 实现，只需要在**解析 + 构造 Proxy** 这一层提供新 API，并在「未知类型」时约定 async 语义。

---

## 方案

### 方案 A：`get_actor(name)` → 动态 Proxy（推荐）

- **API**：`proxy = await pul.get_actor("channel.discord")`，返回一个「无类型」的 `ActorProxy`。
- **实现**：
  - `get_actor(name)` 内部：`ref = await resolve(name)`，然后 `return ActorProxy(ref, method_names=None, async_methods=?)`。
  - 当 `method_names is None` 时，现有逻辑已允许任意方法名，无需改 `__getattr__`。
- **async 语义**（二选一）：
  - **A1**：`async_methods=None` 时**全部按 async 处理**（即 `__getattr__` 里对 `async_methods is None` 时令 `is_async=True`）。
    这样 `await proxy.send_text(...)` 和 `async for x in proxy.generate(...)` 都能用，适用面最大。
  - **A2**：全部按 sync 处理（当前 `async_methods=set()` 的行为）。
    `await proxy.send_text(...)` 仍可用（因为 `_sync_call` 本身是 async），但**流式返回**（async generator）可能只拿到最终结果或行为未定义，需文档说明「流式请用类型化 resolve」。
- **推荐 A1**：实现简单（在 Proxy 里把 `async_methods is None` 视为「全部 async」），且与「未知类型时尽量不限制能力」一致。

**优点**：不增加新类型、不碰协议；调用方只需 `await pul.get_actor(name)` 然后 `await proxy.xxx(...)`，nanobot 可删掉 `get_channel_actor` 和 name→Class 映射。
**缺点**：无类型提示、无静态校验；流式在 A2 下需用类型化 resolve。

---

### 方案 B：`resolve(name)` 返回「默认动态 Proxy」

- 保持 `resolve(name)` 返回 `ActorRef` 的语义，**新增**一个重载或单独函数，例如 `get_proxy(name)` / `actor(name)`，行为同方案 A 的 `get_actor(name)`。
- 或者：为 `ActorRef` 增加 `.proxy(dynamic=True)`，例如 `ref = await resolve(name); proxy = ref.proxy(dynamic=True)`，内部用 `ActorProxy(ref, None, None)` 并约定 async 语义（同 A1/A2）。

**优点**：与现有 `resolve` 返回 `ActorRef` 的语义不冲突；需要底层 ref 时仍用 `resolve`。
**缺点**：多一个 API 或概念（dynamic proxy），需要文档说明与 `SomeClass.resolve` 的差异。

---

### 方案 C：通过元数据 / Describe 协议补全类型信息

- 在 spawn 时把 actor 的「公共方法名 + 是否 async」存到某处（进程内 registry 或通过某 Describe 协议向 actor 查询）。
- `get_actor(name)` 时先 `resolve(name)` 得到 ref，再查元数据得到 `(method_names, async_methods)`，用现有 `ActorProxy(ref, method_names, async_methods)` 构造**类型信息完整**的 Proxy。

**优点**：能区分 sync/async、流式，且不「全部当 async」；理论上可做更好提示。
**缺点**：
- 元数据：若只存在 spawn 端，则跨进程 resolve 时拿不到；若要从 actor 查，需要约定 Describe 协议和实现。
- 实现和运维成本高，对 nanobot 这种「只按名字调几个方法」的场景收益有限。

---

## 建议

- **短期**：采用 **方案 A（`get_actor(name)` + 动态 Proxy）**，并在 Proxy 内采用 **A1**（`async_methods is None` 时全部按 async），这样：
  - 只在一个地方（如 `pulsing.core`）增加 `get_actor(name)`（及可选 `node_id`）。
  - 对 `ActorProxy` 做最小改动：在 `__getattr__` 里当 `self._async_methods is None` 时令 `is_async=True`。
- **命名**：`get_actor(name)` 与现有 `resolve(name)`（返回 ref）区分清晰；若希望更短，可再提供 `pul.actor(name)` 作为别名。
- **文档**：说明「无类型、无补全；流式优先用类型化 resolve」即可。
- 方案 C 可作为后续「可观测性 / 接口发现」的一部分再考虑，不必绑在「按名字解析」的第一版里。

---

## 使用示例（已实现）

```python
# 按名字解析后通过 .as_any() 获得「任意方法转发」的 proxy
import pulsing as pul

ref = await pul.resolve("channel.discord")
proxy = ref.as_any()
await proxy.send_text(chat_id, content)
```

```python
# 或使用独立函数（适用于已有 ActorRef 的场景）
proxy = pul.as_any(ref)
await proxy.send_text(chat_id, content)
```

```python
# 类型化 proxy 也可 .as_any() 得到无类型视图
typed = await SomeClass.resolve("my_actor")
any_proxy = typed.as_any()
await any_proxy.any_method(...)
```
