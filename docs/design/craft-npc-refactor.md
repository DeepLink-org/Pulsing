# Craft 重构：让 Pulsing 做 Pulsing 的事，Craft 只做游戏隐喻

> **核心洞察**：Pulsing 本身就是一个分布式 actor 系统，已内置 actor spawn、gossip 发现、跨节点 RPC、进程隔离、流式传输、supervision 重启、以及 `schedule_self` 尾递归 actor 模式。HubActor 用 600 行代码重新发明了这些能力。重构的本质不是"修 HubActor"，而是**删掉它，让 NPC 直接作为 Pulsing actor 存在，用 self-messaging 实现自主行为链**。
>
> **给实现 AI 的指令**：先通读 `llms.binding.md` 理解 Pulsing API，再动代码。

---

## 1. Pulsing 已经提供了什么

```
┌─────────────────────────────────────────────────────────┐
│                    Pulsing 内置能力                       │
│                                                         │
│  @pul.remote  →  actor 定义、自动 proxy、gossip 注册    │
│  spawn(name=) →  创建 actor、注册到集群                  │
│  resolve(name, cls=) →  跨节点发现 actor                │
│  all_named_actors() →  列出集群中所有 actor             │
│  spawn(new_process=True) →  OS 级进程隔离               │
│  async def → yield  →  自动流式 RPC 响应                │
│  restart_policy  →  actor 挂了自动复活                  │
│  init(addr, seeds) →  加入/创建 gossip 集群             │
│  shutdown()  →  离开集群                                │
│  mailbox  →  自动串行化消息处理                          │
│  schedule_self(msg, delay) → actor 给自己发消息          │
└─────────────────────────────────────────────────────────┘
```

### 1.1 Pulsing → Craft 映射表

| Pulsing API | Craft 概念 | 说明 |
|---|---|---|
| `@pul.remote` | NPC 类定义 | Pulsing 自动生成 proxy，注册到 gossip |
| `NPC.spawn(name="guide", public=True)` | summon NPC | Pulsing 自动注册，其他节点可发现 |
| `pul.resolve("craft/ws/<world>/guide", cls=NPC)` | 找到 NPC | Pulsing gossip 自动发现 |
| `system.all_named_actors()` | `/look` 和 `/who` | Pulsing 列出所有已注册 actor |
| `pul.spawn(FullToolWorker(), new_process=True)` | 工具沙箱 | Pulsing OS 级进程隔离 |
| `async def → yield` | 流式回复 | Pulsing 自动 `async for` 流式传输 |
| `restart_policy="on_failure"` | NPC 复活 | Pulsing supervision 自动重启 |
| `pul.init(addr=..., seeds=...)` | World wake | 加入/创建 gossip 集群 |
| `pul.shutdown()` | World sleep | 离开集群 |
| **`schedule_self(msg, delay)`** | **NPC 自主行为** | **NPC 给自己发消息，驱动自主工作链** |

### 1.2 HubActor 做了什么（全是 Pulsing 已有的事）

| HubActor 做的事 | Pulsing 等价物 | 结论 |
|---|---|---|
| `turn_lock` 串行化 | Pulsing actor mailbox 天然串行 | **多余** |
| `_inbound_agent_messages` 队列 | Pulsing mailbox 自动排队 | **多余** |
| `receive_agent_message` RPC | NPC 方法直接作为 Pulsing RPC | **多余** |
| `CoordinatorRuntime` spawn task | `NPC.spawn(name=...)` | **多余** |
| `ClusterRuntime` resolve + RPC | `pul.resolve()` + 方法调用 | **多余** |
| Coordinator 150ms 轮询等待 | `schedule_self` 自主行为链 | **多余** |
| `call_tool` 工具路由 | NPC 内部方法 | **保留**（craft 特有） |
| `_ensure_worker` / `_respawn_worker` | Pulsing supervision | **可简化** |
| `run_turn` / `run_turn_stream` | NPC 内部方法 | **保留**（LLM turn 是 craft 特有） |

---

## 2. 新架构：Self-Messaging 驱动的自主 NPC

### 2.1 核心模式：NPC 给自己发消息

Pulsing 的 `schedule_self(msg, delay)` 允许 actor 向自己的 mailbox 发送消息。这让 NPC 从"被动等 Player 指令"变成"自主驱动工作链"——这就是"尾递归优化"模式：每个 turn 完成后通过 self-message 触发下一个 turn，不占用调用栈。

```
被动 NPC (HubActor 模式):           自主 NPC (schedule_self 模式):

Player: "fix all tests"            Player: "fix all tests"
  ↓                                  ↓
NPC: turn 1 — 修一个文件            NPC: turn 1 — 跑 pytest, 发现 3 个失败
  ↓                                  ↓ _schedule_self(ContinueWork("fix test_a"))
NPC: 回复 "done"                    NPC: turn 2 — 修 test_a.py
  ↑                                  ↓ _schedule_self(ContinueWork("fix test_b"))
Player 必须再催一次                  NPC: turn 3 — 修 test_b.py
                                     ↓ _schedule_self(ContinueWork("fix test_c"))
                                    NPC: turn 4 — 修 test_c.py
                                     ↓ 工作链结束 → 回复 Player
                                    NPC: 回复 "all done"
```

### 2.2 分层架构

```
┌──────────────────────────────────────────────────────────┐
│                      Craft 层（薄）                        │
│                                                          │
│  NPC 类 (@pul.remote)                                    │
│    ├── say(player, msg)      → 启动工作链                │
│    ├── whisper(npc, msg)     → 同上                      │
│    ├── _schedule_self(item)  → 给自己发消息，接力工作链   │
│    ├── _process_work_loop()  → 尾递归执行引擎            │
│    ├── _run_one_turn(item)   → 跑 LLM turn              │
│    ├── _tool_summon()        → NPC.spawn()              │
│    ├── _tool_whisper()       → pul.resolve() + .whisper()│
│    └── who()                 → 身份卡                    │
│                                                          │
│  AsyncTurnRunner (LLM 大脑)                              │
│  PermissionChecker (权限)    SessionStore (对话存储)      │
│  BuddyState (宠物)           FullToolWorker (隔离工具)    │
│  REPL (对话界面)                                          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                      Pulsing 层                           │
│                                                          │
│  Actor Mailbox 串行化所有消息 (say + self-message)        │
│  schedule_self — actor 给自己发消息，驱动自主行为链       │
│  Remote Spawn / Resolve / RPC / Gossip                   │
│  Process Isolation / Streaming / Supervision             │
└──────────────────────────────────────────────────────────┘
```

---

## 3. NPC 类设计

### 3.0 Self-Messaging 工作链详解

```
say("fix tests")           ← Player 调用，进入 Pulsing mailbox
  └─ _schedule_self(WorkItem(kind="player_say", content="fix tests"))
  └─ 设置 _active_reply_future ← 工作链结束时要回复这个 future

_process_work_loop 取出 WorkItem:
  └─ _run_one_turn:
       └─ runner.run_turn("fix tests")
       └─ LLM: "我先跑 pytest 看看"
       └─ Tool: Bash("pytest")
       └─ LLM: "3 个失败，先修 test_a"
       └─ Tool: Edit("test_a.py", ...)
       └─ LLM: "修好了，继续修 test_b"
       └─ _schedule_self(ContinueWork("fix test_b"))  ← 给自己发消息！

_process_work_loop 取出 ContinueWork:
  └─ _run_one_turn:
       └─ runner.run_turn("[continue] fix test_b")
       └─ ...修 test_b...
       └─ _schedule_self(ContinueWork("fix test_c"))

_process_work_loop 取出 ContinueWork:
  └─ _run_one_turn:
       └─ ...修 test_c...
       └─ LLM: "全部完成"
       └─ 没有更多 self-message → _work_queue 空了
       └─ _maybe_reply(): 检查 _work_queue.empty() → 是
       └─ _active_reply_future.set_result(result)  ← 回复最初调用 say() 的人
```

### 3.1 WorkItem 定义

```python
# 文件: python/pulsing/craft/npc.py (新增，放在 NPC 类之前)

from dataclasses import dataclass, field
import uuid

@dataclass
class WorkItem:
    """工作队列中的一项——可以来自外部 (say/whisper) 或内部 (self-message)。"""
    kind: str                                    # "player_say" | "npc_whisper" | "continue_work" | "summon_notification"
    from_agent: str = ""
    content: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def user_text(self) -> str:
        """转为 runner.run_turn() 的 user 文本。"""
        if self.kind == "player_say":
            return self.content
        elif self.kind == "npc_whisper":
            return f"[{self.from_agent} whispers] {self.content}"
        elif self.kind == "continue_work":
            return f"[continue] {self.content}"
        elif self.kind == "summon_notification":
            return self.content  # XML
        return self.content
```

### 3.2 NPC 完整代码

```python
# 文件: python/pulsing/craft/npc.py

import asyncio, uuid, json, logging, os, xml.sax.saxutils as xml_esc
from typing import Any

import pulsing as pul

from pulsing.craft.npc_class import get_npc_class
from pulsing.craft.runtime.engine_async import AsyncTurnRunner
from pulsing.craft.runtime.permissions import PermissionChecker
from pulsing.craft.runtime.session_store import SessionStore
from pulsing.craft.runtime.companion_buddy import BuddyState
from pulsing.craft.runtime.constants import ISOLATED_TOOL_NAMES
from pulsing.craft.runtime.remote_tool import tool_result_from_worker_value
from pulsing.craft.payload.full_tool_worker import FullToolWorker
from pulsing.craft.cluster.constants import full_agent_name

logger = logging.getLogger(__name__)
_ISOLATED_WORKER_NAME = "craft_isolated_tools"


@pul.remote
class NPC:
    """
    Pulsing actor with LLM brain and self-messaging autonomy.

    Pulsing 提供: gossip 注册、mailbox 串行化、跨节点 RPC、进程隔离、流式响应
    Craft 提供: LLM turn loop、工具系统、权限、session、buddy、self-messaging 工作链
    """

    def __init__(
        self, *,
        name: str, workspace_id: str,
        npc_class: str = "artisan", personality: str = "",
        model: str, provider: str = "anthropic",
        api_key: str | None = None, base_url: str | None = None,
        max_tokens: int = 8192,
        max_summon_depth: int = 3, summon_depth: int = 0,
        tool_denylist: list[str] | None = None,
        auto_approve: bool = False, prompt_callback: Any | None = None,
        sandbox_policy: str = "off", dangerously_disable_sandbox: bool = False,
        cwd: str = ".", resume_id: str | None = None, cost_log: bool = True,
    ) -> None:
        # identity
        self._name = name
        self._workspace_id = workspace_id
        self._cls = get_npc_class(npc_class)
        self._personality = (personality or self._cls.default_personality).strip()

        # LLM
        self._provider = provider; self._model = model
        self._api_key = api_key or os.environ.get(
            "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
        self._base_url = base_url; self._max_tokens = max_tokens

        # summon
        self._max_summon_depth = max_summon_depth
        self._summon_depth = summon_depth
        self._summoned: dict[str, str] = {}

        # tools
        self._tools: dict[str, Any] = {}
        self._tool_denylist = set(tool_denylist or ())
        if summon_depth >= max_summon_depth:
            self._tool_denylist.add("Summon")

        # permissions
        self._checker = PermissionChecker(auto_approve=auto_approve, prompt_callback=prompt_callback)

        # sandbox
        self._sandbox_policy = sandbox_policy
        self._dangerously_disable_sandbox = dangerously_disable_sandbox

        # session
        self._cwd = cwd
        if resume_id:
            self._session = SessionStore(cwd=cwd, model=model, session_id=resume_id)
            self._initial_messages = SessionStore.load_messages(resume_id, cwd)
        else:
            self._session = SessionStore(cwd=cwd, model=model)
            self._initial_messages = []

        # runner
        self._runner: AsyncTurnRunner | None = None

        # === 工作队列 — self-messaging 的载体 ===
        # say()/whisper() 将任务放入队列，_process_work_loop() 逐条取出执行。
        # 每个 turn 结束后，LLM 可通过 _schedule_self(ContinueWork(...))
        # 向队列发消息触发下一个 turn——这就是"尾递归"模式。
        # summon 完成通知也通过 _schedule_self() 注入。
        self._work_queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._work_loop: asyncio.Task | None = None

        # === 当前工作链的 reply future ===
        # say()/whisper() 设置它。工作链最后一个 turn 通过它回复调用者。
        # 当 _work_queue 为空时，说明工作链结束，触发回复。
        self._active_reply_future: asyncio.Future | None = None

        # isolated worker
        self._worker_lock = asyncio.Lock()
        self._worker_handle: pul.IsolatedSpawnHandle | None = None
        self._worker_proxy: pul.ActorProxy | None = None

        # buddy
        self._buddy = BuddyState()
        self._buddy.set_persona_name(self._name)

        # lifecycle
        self._on_start_lock = asyncio.Lock()
        self._on_start_done = False

        # prompt
        self._system_prompt = self._build_prompt()

    # ═══════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════

    async def on_start(self, actor_id) -> None:
        async with self._on_start_lock:
            if self._on_start_done: return
            self._tools = self._build_tools()
            self._runner = self._create_runner()
            await self._ensure_worker()
            self._work_loop = asyncio.create_task(self._process_work_loop())
            self._on_start_done = True
            logger.info("NPC %s [%s] awake depth=%s/%s tools=%s",
                        self._name, self._cls.name, self._summon_depth,
                        self._max_summon_depth, len(self._tools))

    async def on_stop(self) -> None:
        if self._work_loop:
            self._work_loop.cancel()
            try: await self._work_loop
            except asyncio.CancelledError: pass
        if self._worker_handle is not None:
            proc = self._worker_handle.process
            if proc.returncode is None: proc.terminate()

    # ═══════════════════════════════════════════
    # Self-Messaging — NPC 自主性的核心
    # ═══════════════════════════════════════════

    async def _schedule_self(self, item: WorkItem) -> None:
        """给自己发消息——Pulsing schedule_self 的 Python 等价实现。"""
        await self._work_queue.put(item)

    async def _process_work_loop(self) -> None:
        """尾递归执行引擎——逐条取出 WorkItem，跑 LLM turn。"""
        while True:
            item = await self._work_queue.get()
            await self._run_one_turn(item)

    async def _run_one_turn(self, item: WorkItem) -> None:
        """跑一个 LLM turn。完成后检查队列是否为空——空则回复调用者。"""
        try:
            result = await self._runner.run_turn(item.user_text())
            self._buddy.on_turn_finished(
                ok=result.get("ok", True),
                had_tool_calls=any(e.get("kind") == "tool_call"
                                   for e in result.get("events", [])))
            # 如果队列空了，说明工作链结束
            if self._work_queue.empty() and self._active_reply_future:
                fut = self._active_reply_future
                self._active_reply_future = None
                if not fut.done(): fut.set_result(result)
        except asyncio.CancelledError:
            self._fail_reply(asyncio.CancelledError("work cancelled"))
        except Exception as e:
            logger.exception("NPC %s turn failed", self._name)
            self._fail_reply(e)

    def _fail_reply(self, exc: BaseException) -> None:
        if self._active_reply_future and not self._active_reply_future.done():
            self._active_reply_future.set_exception(exc)
            self._active_reply_future = None

    # ═══════════════════════════════════════════
    # 公网 API — 入口点
    # ═══════════════════════════════════════════

    async def say(self, from_name: str, message: str, *,
                  wait: bool = True, timeout: float = 600.0) -> dict[str, Any]:
        """Player 对 NPC 说话。启动工作链。"""
        body = (message or "").strip()
        if not body: return {"ok": False, "error": "empty message"}
        item = WorkItem(kind="player_say", from_agent=from_name, content=body)
        if wait:
            loop = asyncio.get_event_loop()
            self._active_reply_future = loop.create_future()
            await self._schedule_self(item)
            try: return await asyncio.wait_for(self._active_reply_future, timeout=timeout)
            except asyncio.TimeoutError:
                self._active_reply_future = None
                return {"ok": False, "error": f"{self._name} timeout"}
        else:
            await self._schedule_self(item)
            return {"ok": True, "accepted": True}

    async def whisper(self, from_name: str, message: str, *,
                      wait: bool = True, timeout: float = 600.0) -> dict[str, Any]:
        """其他 NPC 对此 NPC 私聊。启动工作链。"""
        body = (message or "").strip()
        if not body: return {"ok": False, "error": "empty message"}
        item = WorkItem(kind="npc_whisper", from_agent=from_name, content=body)
        if wait:
            loop = asyncio.get_event_loop()
            self._active_reply_future = loop.create_future()
            await self._schedule_self(item)
            try: return await asyncio.wait_for(self._active_reply_future, timeout=timeout)
            except asyncio.TimeoutError:
                self._active_reply_future = None
                return {"ok": False, "error": f"{self._name} timeout"}
        else:
            await self._schedule_self(item)
            return {"ok": True, "accepted": True}

    def who(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "full_name": full_agent_name(self._name, workspace_id=self._workspace_id),
            "class": self._cls.name, "class_description": self._cls.description,
            "personality": self._personality[:200], "workspace_id": self._workspace_id,
            "summon_depth": self._summon_depth, "max_summon_depth": self._max_summon_depth,
            "session_id": self._session.session_id, "cwd": self._cwd,
            "model": self._model, "provider": self._provider,
            "buddy": self._buddy.status_line(),
        }

    def get_session_id(self) -> str:
        return self._session.session_id

    # ═══════════════════════════════════════════
    # Tool Backend
    # ═══════════════════════════════════════════

    async def call_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        if name == "Summon":    return await self._tool_summon(**kwargs)
        if name == "Whisper":   return await self._tool_whisper(**kwargs)
        if name == "ListNPCs":  return await self._tool_list_npcs(**kwargs)
        if name in ISOLATED_TOOL_NAMES: return await self._isolated_tool(name, kwargs)
        tool = self._tools.get(name)
        if tool is None:
            from pulsing.craft.runtime.tool_base import ToolResult
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        return await asyncio.to_thread(tool.execute, **kwargs)

    # ═══════════════════════════════════════════
    # Summon — 使用 Pulsing 原生 spawn
    # ═══════════════════════════════════════════

    async def _tool_summon(self, **kwargs: Any) -> Any:
        from pulsing.craft.runtime.tool_base import ToolResult
        goal = str(kwargs.get("goal") or kwargs.get("task") or "").strip()
        if not goal: return ToolResult(content="Summon: goal required.", is_error=True)
        npc_class = str(kwargs.get("npc_class") or "artisan").strip()
        name = str(kwargs.get("name") or "").strip() or f"sub-{uuid.uuid4().hex[:6]}"
        task_id = str(kwargs.get("task_id") or "").strip() or f"s-{uuid.uuid4().hex[:10]}"

        handle = await NPC.spawn(
            name=name, workspace_id=self._workspace_id,
            npc_class=npc_class, personality=kwargs.get("personality", ""),
            model=kwargs.get("model") or self._model, provider=self._provider,
            max_summon_depth=self._max_summon_depth, summon_depth=self._summon_depth + 1,
            auto_approve=self._checker._auto_approve, sandbox_policy=self._sandbox_policy,
            cwd=self._cwd,
        )
        self._summoned[task_id] = full_agent_name(name, workspace_id=self._workspace_id)
        proxy = pul.ActorProxy(handle.ref, NPC._methods, NPC._async_methods)
        asyncio.create_task(self._await_summoned(task_id, goal, proxy, name))
        return ToolResult(content=json.dumps({
            "task_id": task_id, "npc": name, "class": npc_class,
            "status": "summoned", "goal": goal,
        }, ensure_ascii=False))

    async def _await_summoned(self, task_id: str, goal: str, proxy: Any, name: str) -> None:
        try:
            result = await proxy.whisper(from_name=self._name, message=goal, wait=True)
            status = "completed" if result.get("ok") else "failed"
            text = result.get("assistant_text", "")[:8000]
            # ★ 通过 self-message 注入通知，而不是直接修改消息历史
            notif = _format_summon_result(task_id, name, status, text[:2000], text)
            await self._schedule_self(WorkItem(kind="summon_notification", content=notif))
        except Exception as e:
            notif = _format_summon_result(task_id, name, "failed", str(e)[:2000], "")
            await self._schedule_self(WorkItem(kind="summon_notification", content=notif))
        finally:
            self._summoned.pop(task_id, None)

    # ═══════════════════════════════════════════
    # Whisper — 使用 Pulsing 原生 resolve + RPC
    # ═══════════════════════════════════════════

    async def _tool_whisper(self, **kwargs: Any) -> Any:
        from pulsing.craft.runtime.tool_base import ToolResult
        target = str(kwargs.get("to") or kwargs.get("npc") or "").strip()
        message = str(kwargs.get("message") or "").strip()
        if not target: return ToolResult(content="Whisper: to/npc required.", is_error=True)
        if not message: return ToolResult(content="Whisper: message required.", is_error=True)
        wait = bool(kwargs.get("wait", True))
        timeout = float(kwargs.get("timeout", 600.0))
        full = full_agent_name(target, workspace_id=self._workspace_id)
        try:
            peer = await pul.resolve(full, cls=NPC, timeout=min(timeout, 120.0))
        except Exception as e:
            return ToolResult(content=f"Cannot find '{target}': {e}", is_error=True)
        try:
            result = await peer.whisper(from_name=self._name, message=message,
                                        wait=wait, timeout=timeout)
        except Exception as e:
            return ToolResult(content=f"Whisper failed: {e}", is_error=True)
        if wait and isinstance(result, dict):
            text = result.get("assistant_text") or result.get("error") or str(result)
            return ToolResult(content=text[:8000])
        return ToolResult(content=str(result))

    async def _tool_list_npcs(self, **kwargs: Any) -> Any:
        from pulsing.craft.runtime.tool_base import ToolResult
        system = pul.get_system()
        try: all_actors = await system.all_named_actors()
        except Exception as e: return ToolResult(content=f"Failed: {e}", is_error=True)
        ws_prefix = f"craft/ws/{self._workspace_id}/"
        lines = []
        for info in all_actors:
            path = str(info.get("path", ""))
            if path.startswith("actors/"): path = path[7:]
            if path.startswith(ws_prefix):
                name = path[len(ws_prefix):]
                node = str(info.get("node_id", "?"))[:12]
                lines.append(f"  {name} (node={node})")
        if not lines: return ToolResult(content="(no other NPCs)")
        return ToolResult(content="NPCs:\n" + "\n".join(lines))

    # ═══════════════════════════════════════════
    # Isolated Worker
    # ═══════════════════════════════════════════

    async def _isolated_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        from pulsing.craft.runtime.tool_base import ToolResult
        last_exc = None
        for attempt in range(2):
            try:
                await self._ensure_worker()
                raw = await getattr(self._worker_proxy, name)(**kwargs)
                return tool_result_from_worker_value(raw)
            except BaseException as e:
                last_exc = e
                async with self._worker_lock:
                    await self._respawn_worker(f"recover after {name}")
        return ToolResult(content=f"Tool failed: {last_exc!r}", is_error=True)

    async def _ensure_worker(self) -> None:
        async with self._worker_lock:
            if self._worker_handle is not None and self._worker_handle.process.returncode is None:
                return
            await self._respawn_worker("worker missing")

    async def _respawn_worker(self, reason: str) -> None:
        if self._worker_handle is not None:
            proc = self._worker_handle.process
            if proc.returncode is None:
                proc.terminate()
                try: await asyncio.wait_for(proc.wait(), timeout=8.0)
                except asyncio.TimeoutError: proc.kill(); await proc.wait()
            self._worker_handle = None; self._worker_proxy = None
        logger.info("NPC %s: spawning worker (%s)", self._name, reason)
        h = await pul.spawn(
            FullToolWorker(sandbox_policy=self._sandbox_policy,
                           dangerously_disable_sandbox=self._dangerously_disable_sandbox),
            new_process=True, name=_ISOLATED_WORKER_NAME, public=False, restart_policy="never",
        )
        if not isinstance(h, pul.IsolatedSpawnHandle):
            raise TypeError("expected IsolatedSpawnHandle")
        self._worker_handle = h
        self._worker_proxy = pul.ActorProxy(h.ref, FullToolWorker._methods, FullToolWorker._async_methods)

    # ═══════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════

    def _build_prompt(self) -> str:
        cls = self._cls
        parts = [
            f"You are NPC '{self._name}' in world '{self._workspace_id}'.",
            f"Class: {cls.name} — {cls.description}",
            f"Personality: {self._personality}",
            "",
            "You live in a game-like dev world. The Player may /say things to you.",
            "You can Summon NPCs to delegate work and Whisper to coordinate.",
            "Use tools to work on tasks autonomously. When a task requires multiple steps,",
            "complete each step, then the system will continue to the next automatically.",
            "You don't need to finish everything in one turn — use tools step by step.",
            "",
            f"Working directory: {self._cwd}",
        ]
        if cls.prompt_extra: parts.append(f"\n{cls.prompt_extra}")
        return "\n".join(parts)

    def _create_runner(self) -> AsyncTurnRunner:
        runner = AsyncTurnRunner(
            self, self._tools, self._checker,
            provider=self._provider, model=self._model,
            api_key=self._api_key, base_url=self._base_url,
            system_prompt=self._system_prompt,
            session_store=self._session,
            stream_assistant=False, text_stream_callback=None,
            usage_observer=self._on_usage if self._cost_log_enabled else None,
        )
        if self._initial_messages: runner.set_messages(self._initial_messages)
        return runner

    def _build_tools(self) -> dict[str, Any]:
        from pulsing.craft.npc_tools import SummonTool, WhisperTool, ListNPCsTool
        from pulsing.craft.runtime.tools_pkg import (
            ReadTool, GlobTool, GrepTool, EditTool, WriteTool, BashTool,
            MemoryAppendTool, MemorySearchTool, SkillRunTool, SkillsListTool,
            McpCatalogTool, FetchUrlTool, AskUserQuestionTool, BuddyStatusTool,
            EnterDreamModeTool, ExitDreamModeTool, EnterPlanModeTool, ExitPlanModeTool,
        )
        all_tools = {
            "Read": ReadTool, "Glob": GlobTool, "Grep": GrepTool,
            "Edit": EditTool, "Write": WriteTool, "Bash": BashTool,
            "MemoryAppend": lambda: MemoryAppendTool(self._cwd),
            "MemorySearch": lambda: MemorySearchTool(self._cwd),
            "SkillRun": lambda: SkillRunTool(self._cwd), "SkillsList": SkillsListTool,
            "McpCatalog": McpCatalogTool, "FetchUrl": FetchUrlTool,
            "AskUserQuestion": AskUserQuestionTool,
            "BuddyStatus": lambda: BuddyStatusTool(self._buddy),
            "EnterDreamMode": lambda: EnterDreamModeTool(self._checker),
            "ExitDreamMode": lambda: ExitDreamModeTool(self._checker),
            "EnterPlanMode": lambda: EnterPlanModeTool(self._checker),
            "ExitPlanMode": lambda: ExitPlanModeTool(self._checker),
            "Summon": SummonTool, "Whisper": WhisperTool,
            "ListNPCs": lambda: ListNPCsTool(self._workspace_id),
        }
        allow = set(self._cls.default_tools)
        forbid = set(self._cls.forbidden_tools) | self._tool_denylist
        tools = {}
        for name, factory in all_tools.items():
            if name in forbid: continue
            if allow and name not in allow: continue
            tools[name] = factory() if callable(factory) else factory()
        return tools

    def _on_usage(self, usage: dict[str, int]) -> None:
        try:
            from pulsing.craft.runtime.cost_log import append_usage
            append_usage(self._session.session_dir, self._session.session_id,
                         model=self._model, usage=usage)
        except Exception: pass

    @property
    def _cost_log_enabled(self) -> bool:
        return True  # 后续可配置


def _format_summon_result(task_id: str, npc_name: str, status: str,
                          summary: str, result: str) -> str:
    e = xml_esc.escape
    parts = [
        "<summon-result>",
        f"<task-id>{e(task_id)}</task-id>",
        f"<npc>{e(npc_name)}</npc>",
        f"<status>{e(status)}</status>",
        f"<summary>{e(summary)}</summary>",
    ]
    if result: parts.append(f"<result>{e(result)}</result>")
    parts.append("</summon-result>")
    return "\n".join(parts)
```

---

## 4. 支持文件

### 4.1 `npc_class.py`

```python
from dataclasses import dataclass, field

@dataclass
class NPCClass:
    name: str; description: str
    default_personality: str = ""
    prompt_extra: str = ""
    default_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)

_registry: dict[str, NPCClass] = {}

def register(cls): _registry[cls.name] = cls; return cls
def get_npc_class(name): return _registry.get(name, _registry["artisan"])
def list_classes(): return sorted(_registry.keys())

register(NPCClass(name="artisan",
    description="工匠 — 读写文件、执行命令。默认 NPC。",
    default_personality="helpful and precise",
    default_tools=["Read","Glob","Grep","Edit","Write","Bash",
                   "MemoryAppend","MemorySearch","SkillRun","SkillsList",
                   "McpCatalog","FetchUrl","AskUserQuestion","BuddyStatus",
                   "EnterDreamMode","ExitDreamMode","EnterPlanMode","ExitPlanMode"]))

register(NPCClass(name="quest_giver",
    description="任务发布者 — 召唤 NPC、分配任务、协调团队。",
    default_personality="organized and strategic",
    prompt_extra="You can Summon NPCs to delegate and Whisper to coordinate.",
    default_tools=["Summon","Whisper","ListNPCs","Read","Glob","Grep",
                   "MemoryAppend","MemorySearch","AskUserQuestion","BuddyStatus",
                   "EnterPlanMode","ExitPlanMode","SkillsList","McpCatalog"]))

register(NPCClass(name="scholar",
    description="学者 — 只读工具，代码审查和分析。",
    default_personality="critical and detail-oriented",
    default_tools=["Read","Glob","Grep","MemoryAppend","MemorySearch",
                   "AskUserQuestion","BuddyStatus","SkillsList","McpCatalog","FetchUrl",
                   "EnterDreamMode","ExitDreamMode"],
    forbidden_tools=["Edit","Write","Bash"]))

register(NPCClass(name="oracle",
    description="先知 — 信息收集，FetchUrl 和搜索。",
    default_personality="curious and resourceful",
    default_tools=["Read","Glob","Grep","FetchUrl","MemoryAppend","MemorySearch",
                   "AskUserQuestion","BuddyStatus","SkillsList","McpCatalog"],
    forbidden_tools=["Edit","Write","Bash","Summon","Whisper"]))
```

### 4.2 `npc_tools.py`

```python
from typing import Any
from pulsing.craft.runtime.tool_base import Tool, ToolResult
from pulsing.craft.runtime.tools_pkg import _json_schema_object

class SummonTool(Tool):
    @property
    def name(self): return "Summon"
    @property
    def description(self): return "Summon another NPC to help."
    @property
    def input_schema(self): return _json_schema_object({
        "goal": {"type": "string"},
        "npc_class": {"type": "string", "enum": ["artisan","scholar","oracle","quest_giver"]},
        "name": {"type": "string"}, "personality": {"type": "string"},
        "task_id": {"type": "string"},
    }, ["goal"])
    def is_read_only(self): return True
    def execute(self, **kw): raise RuntimeError("via NPC._tool_summon")

class WhisperTool(Tool):
    @property
    def name(self): return "Whisper"
    @property
    def description(self): return "Send a message to another NPC."
    @property
    def input_schema(self): return _json_schema_object({
        "to": {"type": "string"}, "npc": {"type": "string"},
        "message": {"type": "string"},
        "wait": {"type": "boolean"}, "timeout": {"type": "number"},
    }, ["message"])
    def is_read_only(self): return True
    def execute(self, **kw): raise RuntimeError("via NPC._tool_whisper")

class ListNPCsTool(Tool):
    def __init__(self, workspace_id): self._ws = workspace_id
    @property
    def name(self): return "ListNPCs"
    @property
    def description(self): return "List NPCs in this world."
    @property
    def input_schema(self): return _json_schema_object({}, [])
    def is_read_only(self): return True
    def execute(self, **kw): raise RuntimeError("via NPC._tool_list_npcs")
```

---

## 5. 与 HubActor 对比

| 维度 | HubActor | NPC |
|---|---|---|
| Actor 注册 | `HubActor.spawn(name=, public=True)` | `NPC.spawn(name=, public=True)` — **一样** |
| 消息串行化 | `turn_lock` | Pulsing mailbox + 内部 work queue |
| 跨节点通信 | `ClusterRuntime` → `pul.resolve` → `receive_agent_message` | `pul.resolve` → `peer.whisper()` — **直接 RPC** |
| 召唤 NPC | `CoordinatorRuntime` → asyncio.Task | `NPC.spawn()` — **Pulsing 原生** |
| **自主行为** | **Coordinator 150ms 轮询** | **`_schedule_self` 尾递归工作链** |
| summon 通知注入 | `enqueue_coordinator_notification` → 等 hub drain | `_schedule_self(WorkItem)` — 作为普通 turn 执行 |
| 进程隔离 | `pul.spawn(new_process=True)` | `pul.spawn(new_process=True)` — **一样** |
| 代码量 | ~600 行 + coordinator ~250 行 | ~300 行 |

---

## 6. 文件变更

### 删除（Claude Code 遗留）
```
runtime/hub_actor.py, runtime/coordinator.py, runtime/cluster_tools.py,
runtime/split_tools.py, runtime/run_hub.py, runtime/slash_hub.py,
runtime/tui_app.py, cluster/messaging.py,
cluster/run_agent.py, cluster/run_cluster.py, cluster/run_ctl.py,
cluster/controller_repl.py, cluster/cli_common.py
```

### 新增
```
npc.py           ★ 核心 NPC 类 (~300 行)
npc_class.py     ★ NPC 职业定义
npc_tools.py     ★ Summon/Whisper/ListNPCs 工具 schema
repl_npc.py      ★ NPC 对话 REPL
```

### 保留（Pulsing 不提供的）
```
runtime/engine_async.py, engine.py, tool_base.py, tools_impl.py, tools_pkg.py,
permissions.py, sandbox.py, sandbox_manager.py,
session_store.py, memory_kairos.py, companion_buddy.py,
compact.py, llm_client.py, cost_log.py, cost_estimate.py,
mcp_catalog.py, skills_registry.py, stubs_info.py, remote_tool.py, constants.py,
payload/full_tool_worker.py, payload/tool_actor.py,
cluster/constants.py, cluster/discovery.py,
workspace/, commands/
```

---

## 7. 实现步骤

1. `npc_class.py` — 无依赖
2. `npc_tools.py` — 依赖 tool_base + tools_pkg
3. `npc.py` + `WorkItem` — 核心，依赖以上 + engine_async 等
4. `repl_npc.py` — NPC 对话 REPL
5. 更新 `helpers.py` — `spawn_npc` 改用 `NPC.spawn`
6. 更新 `commands/npc.py` — `run_say` 改用 `peer.say()`
7. 简化 `__main__.py` — 去掉 hub/cluster/agent/ctl
8. 删除遗留文件 + 更新测试

---

## 8. 验收标准

- [ ] `pcraft init && pcraft wake` — World 启动
- [ ] `pcraft npc summon coder --class artisan` — summon NPC
- [ ] `pcraft npc who` — 列出 NPC（使用 `all_named_actors()`）
- [ ] `pcraft npc say guide "fix tests"` — Player 对 NPC 说话
- [ ] NPC 使用 Summon 工具召唤其他 NPC（使用 `NPC.spawn()`）
- [ ] NPC 使用 Whisper 工具与其他 NPC 通信（使用 `pul.resolve()` + RPC）
- [ ] **NPC 通过 `_schedule_self` 自主驱动多 turn 工作链**
- [ ] summon 完成通知通过 `_schedule_self` 注入（而非 HubActor 的 enqueue + drain）
- [ ] Summon depth 限制生效
- [ ] 不同 class 的 NPC 有不同的工具集
- [ ] `pcraft sleep` — World 休眠
