"""
Agent 装饰器 - 带元信息的 @remote

功能上等同于 @remote，但额外携带元信息：
- role: 角色名称
- goal: 目标描述
- backstory: 背景故事
- tags: 自定义标签

这些元信息可用于：
1. 运行时可视化（Agent 拓扑图）
2. 调试和日志
3. 自动生成文档
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pulsing.actor import remote

T = TypeVar("T")

# 全局元信息注册表: actor_name -> AgentMeta
_agent_meta_registry: dict[str, "AgentMeta"] = {}


@dataclass
class AgentMeta:
    """Agent 元信息"""

    role: str = ""
    goal: str = ""
    backstory: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        return f"{self.role}" if self.role else "Agent"


def agent(
    role: str = "",
    goal: str = "",
    backstory: str = "",
    **tags: Any,
) -> Callable[[type[T]], type[T]]:
    """
    Agent 装饰器 - 等同于 @remote，但附加元信息

    元信息通过 get_agent_meta(name) 获取。

    Example:
        @agent(role="研究员", goal="深入分析")
        class Researcher:
            async def analyze(self, topic: str) -> str:
                ...

        async with runtime():
            r = await Researcher.spawn(name="researcher")

            # 获取元信息
            meta = get_agent_meta("researcher")
            print(meta.role)  # "研究员"
    """

    def decorator(cls: type[T]) -> type[T]:
        # 创建元信息
        meta = AgentMeta(role=role, goal=goal, backstory=backstory, tags=tags)

        # 存储到类上（供 spawn 时使用）
        cls._agent_meta_template = meta  # type: ignore

        # 包装 spawn 方法来注册元信息
        actor_cls = remote(cls)
        original_spawn = actor_cls.spawn

        async def spawn_with_meta(*args: Any, name: str | None = None, **kwargs: Any):
            proxy = await original_spawn(*args, name=name, **kwargs)
            # 注册元信息
            if name:
                _agent_meta_registry[name] = meta
            return proxy

        actor_cls.spawn = spawn_with_meta
        actor_cls.__agent_meta__ = meta  # 类级别也可访问

        return actor_cls

    return decorator


def get_agent_meta(name: str) -> AgentMeta | None:
    """根据 Actor 名称获取元信息"""
    return _agent_meta_registry.get(name)


def list_agents() -> dict[str, AgentMeta]:
    """列出所有已注册的 Agent 及其元信息"""
    return _agent_meta_registry.copy()


def clear_agent_registry() -> None:
    """清除注册表（用于测试）"""
    _agent_meta_registry.clear()
