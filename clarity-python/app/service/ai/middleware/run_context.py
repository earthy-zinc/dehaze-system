"""Agent 运行时上下文（run-scoped）

图按 (agent_id, version_no, model_id) 缓存复用，构图时创建的 ctx dict 与
middleware/工具闭包绑定；若跨 run 共享同一 dict，并发推理会互相污染
（user_id/计费上下文/stream_session_id/任务状态错乱，属跨用户数据串扰）。

因此运行时上下文改为：每次 run 的驱动任务（reasoning_service.run/resume，
即 astream 的调用方）经 ensure_run_ctx 预置独立 dict，图内全部节点任务经
asyncio 上下文拷贝共享同一 dict（节点内变更全局可见）。

**为什么不能在 abefore_agent 里创建**：langchain factory 将 before_agent 挂为
独立图节点（factory.py: `{name}.before_agent`），LangGraph 节点在各自独立的
asyncio 任务中执行——abefore_agent 内 set_run_ctx 仅在该节点任务内可见，
model/tools/end 节点会回退到图缓存的共享模板（跨 run 串扰、计费上下文丢失、
step_count 跨 run 累积泄漏）。

构图时传入的 ctx 仅作静态配置模板（护栏参数），在无 run 上下文的场景
（如单测直调工具/middleware）作回退值。
"""

from contextlib import contextmanager
from contextvars import ContextVar, Token

_run_ctx: ContextVar[dict | None] = ContextVar("ai_agent_run_ctx", default=None)


def ensure_run_ctx(template: dict) -> dict:
    """在图运行驱动任务中预置本 run 上下文（run/resume astream 前调用）。

    已预置（同一 run 内重复调用）时直接复用；并发 run 各自在独立驱动任务中
    预置，互不可见，天然隔离。
    """
    ctx = _run_ctx.get()
    if ctx is None:
        ctx = dict(template)
        _run_ctx.set(ctx)
    return ctx


# 子 Agent run 标记：deepagents task 工具执行期间置为本次调用的实例 id（嵌套子图继承），
# 供计费/采集钩子区分主/子口径（主图预扣-结算，子图实报实销）；
# 写冲突仲裁再据此区分"同一子 Agent 被并行派发的多个实例"（实例相同才算重入）。
_subagent_instance: ContextVar[str | None] = ContextVar("ai_agent_subagent_instance", default=None)


def get_run_ctx() -> dict | None:
    """读取当前 run 的运行时上下文，无则返回 None"""
    return _run_ctx.get()


def set_run_ctx(ctx: dict) -> None:
    """绑定当前 run 的运行时上下文（abefore_agent 入口调用）"""
    _run_ctx.set(ctx)


def in_subagent_run() -> bool:
    """当前是否处于子 Agent run 内（task 工具执行期间）"""
    return _subagent_instance.get() is not None


def current_subagent_instance() -> str | None:
    """当前子 Agent 实例 id（主 run 为 None）"""
    return _subagent_instance.get()


def enter_subagent_run(instance_id: str) -> Token:
    """标记进入子 Agent run（awrap_tool_call 拦截 task 工具时调用），返回 token 供退出还原"""
    return _subagent_instance.set(instance_id)


def exit_subagent_run(token: Token) -> None:
    """退出子 Agent run 标记（嵌套子图逐层还原）"""
    _subagent_instance.reset(token)


def current_run_ctx(default: dict | None = None) -> dict:
    """取当前 run 上下文；未初始化时回退 default（模板），两者皆无则显式报错"""
    ctx = _run_ctx.get()
    if ctx is not None:
        return ctx
    if default is None:
        raise RuntimeError("Agent 运行时上下文未初始化（须在图 run 内访问）")
    return default


# 子智能体标识：子 Agent 模型调用期间绑定其身份（agentCode），供用量按运行期上下文归属；
# 主 Agent 调用处为 None（不归属）。与 _subagent_instance 的区别：后者标识"哪一次派发实例"
# （写冲突仲裁用），本变量标识"哪个子 Agent"（用量分类用），两者正交可叠加。
_subagent_code: ContextVar[str | None] = ContextVar("ai_agent_subagent_code", default=None)


@contextmanager
def subagent_scope(code: str | None):
    """子 Agent 模型调用期间绑定子智能体标识（主 Agent 传 None 时不生效）。

    子 Agent 的 DehazeHooksMiddleware.awrap_model_call 包裹模型调用时进入本作用域，
    采集层据此在内存按 agentCode 聚合用量；作用域退出还原（嵌套子 Agent 逐层还原）。
    """
    if not code:
        yield
        return
    token = _subagent_code.set(code)
    try:
        yield
    finally:
        _subagent_code.reset(token)


def current_subagent_code() -> str | None:
    """当前子智能体标识（主 Agent 调用处为 None）"""
    return _subagent_code.get()
