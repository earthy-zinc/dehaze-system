"""可观测性采集器（F-M08-013 数据层与采集链路）

双粒度旁路采集：
- 每次模型调用（span 级）：llm_client 包装层经 begin_llm_call 记录一条
  sys_ai_llm_call（输入构成/输出摘要/usage/首 Token/耗时/状态），写库失败仅告警。
- 消息级汇总：推理入口 start 创建采集器并记录上下文构成快照（context_snapshot），
  推理结束（after_agent 钩子或推理编排的失败/中断收尾）经 settle 聚合写入
  sys_ai_trace（幂等，按 trace_id 唯一）。

采集器经 ContextVar 请求级隔离：trace_id 复用日志链路 trace_id（无请求上下文时
降级生成）。主对话链路由推理入口 start 开启；摘要/记忆提取/建议/步骤摘要等
旁路 LLM 调用经 bypass_span 采集为独立过程链（trace_type 区分）。采集链路任何
失败不得影响对话主链路与旁路业务行为。
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from app.core.exceptions import BusinessException
from app.core.result import _get_trace_id
from app.database import get_db_session
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository

logger = logging.getLogger(__name__)

# 过程链状态（sys_ai_trace.status）：1成功/2失败/3中断/4超时
TRACE_STATUS_SUCCESS = 1
TRACE_STATUS_FAILED = 2
TRACE_STATUS_INTERRUPTED = 3
TRACE_STATUS_TIMEOUT = 4
# LLM 调用状态（sys_ai_llm_call.status）：1成功/2失败/3超时
CALL_STATUS_SUCCESS = 1
CALL_STATUS_FAILED = 2
CALL_STATUS_TIMEOUT = 3

# 输出摘要截断长度（纯技术参数，设计 §3 固化为代码常量）
OUTPUT_SNAPSHOT_MAX_CHARS = 500
# 系统提示正文快照截断上限（审计回放需正文，防超长提示撑爆快照）
SYSTEM_PROMPT_MAX_CHARS = 5000
# input_snapshot 每条消息原文截断上限（审计回放需完整输入消息，防超长内容撑爆快照）
MESSAGE_CONTENT_MAX_CHARS = 2000
# memory 构成项每条注入记忆原文截断上限
MEMORY_CONTENT_MAX_CHARS = 1000
# 工具描述截断上限
TOOL_DESC_MAX_CHARS = 500
# 工具调用参数摘要截断长度
_TOOL_ARGS_MAX_CHARS = 200


def _estimate_tokens(text: str | None) -> int:
    """token 估算（字符数/4，与 LlmClient.count_tokens 同口径，空文本为 0）"""
    return max(1, len(text) // 4) if text else 0


def error_type_of(exc: BaseException) -> str:
    """异常 → 过程链 error_type（varchar(32)，业务异常取错误码，其余取异常类名）"""
    if isinstance(exc, BusinessException):
        return str(exc.code.code)[:32]
    return type(exc).__name__[:32]


class TraceCollector:
    """单次推理的过程链采集器（请求级，经 ContextVar 隔离）"""

    def __init__(
        self,
        trace_id: str,
        *,
        conversation_id: int,
        message_id: int,
        user_id: int | None,
        agent_code: str | None,
        model_id: str | None,
    ) -> None:
        self.trace_id = trace_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.user_id = user_id
        self.agent_code = agent_code
        self.model_id = model_id
        self.trace_type = "conversation"
        self._started = time.perf_counter()
        self._seq = 0
        self._step_position = 0
        self._settled = False
        # 上下文构成快照（§2.2）与压缩/截断事件
        self.context_items: list[dict] = []
        self.context_events: list[dict] = []
        # LLM 调用聚合（usage 缺失时的兜底口径）
        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.first_token_ms: int | None = None

    # ── 上下文快照（§2.2）─────────────────────────────

    def record_context(
        self,
        *,
        system_prompt: str | None,
        messages: list[dict],
        injected_memories: list[dict] | None,
        summary: str | None,
    ) -> None:
        """推理入口记录上下文构成：系统提示/会话摘要/历史消息(按角色计数)/注入记忆。

        仅记录当前实现实际注入的构成项；工具定义与工具结果在 LLM 调用级
        input_snapshot/tool_call 逐轮采集。
        """
        if system_prompt:
            self.context_items.append(
                {
                    "type": "system",
                    "tokens": _estimate_tokens(system_prompt),
                    "content": system_prompt[:SYSTEM_PROMPT_MAX_CHARS],
                }
            )
        if summary:
            tokens = _estimate_tokens(summary)
            self.context_items.append(
                {
                    "type": "summary",
                    "tokens": tokens,
                    "source": "summarized",
                    "content": summary[:SYSTEM_PROMPT_MAX_CHARS],
                }
            )
            self.context_events.append({"event": "summarize", "tokens": tokens})
        counts = {"user": 0, "assistant": 0, "tool": 0}
        history_tokens = 0
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue  # system 注入（摘要/记忆块）由独立构成项承载，不计入历史
            if role in counts:
                counts[role] += 1
            history_tokens += _estimate_tokens(msg.get("content"))
        if counts:
            self.context_items.append(
                {"type": "history", "counts": counts, "tokens": history_tokens, "source": "raw"}
            )
        if injected_memories:
            memory_tokens = sum(_estimate_tokens(m.get("content")) for m in injected_memories)
            self.context_items.append(
                {
                    "type": "memory",
                    "count": len(injected_memories),
                    "tokens": memory_tokens,
                    "items": [
                        {
                            "memory_id": m.get("memory_id"),
                            "memory_type": m.get("memory_type"),
                            "source": m.get("source"),
                            "content": (m.get("content") or "")[:MEMORY_CONTENT_MAX_CHARS],
                        }
                        for m in injected_memories
                    ],
                }
            )

    # ── LLM 调用级采集（§2.3）─────────────────────────

    def begin_llm_call(
        self,
        model_id: str,
        messages: list[dict],
        system_prompt: str | None,
        tools: list[dict] | None,
    ) -> "LlmCallRecord":
        self._seq += 1
        return LlmCallRecord(
            self,
            seq=self._seq,
            step_position=self._step_position or None,
            model_id=model_id,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        )

    def _aggregate_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        first_token_ms: int | None,
    ) -> None:
        self.llm_call_count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens
        if first_token_ms is not None and (
            self.first_token_ms is None or first_token_ms < self.first_token_ms
        ):
            self.first_token_ms = first_token_ms

    # ── 消息级汇总落盘（§2.4）─────────────────────────

    async def settle(
        self,
        *,
        status: int,
        error_type: str | None = None,
        error_detail: Any | None = None,
        usage: dict | None = None,
        step_count: int = 0,
        actual_model: str | None = None,
    ) -> None:
        """聚合写入 sys_ai_trace（幂等，重复结算自动跳过）。

        消耗优先取计费口径 usage（含多模态归集），缺失时回退 LLM 调用聚合。
        """
        if self._settled:
            return
        self._settled = True
        usage = usage or {}
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or self.prompt_tokens
        completion = (
            usage.get("output_tokens") or usage.get("completion_tokens") or self.completion_tokens
        )
        cached = usage.get("cached_input_tokens", self.cached_tokens)
        values = {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "agent_code": self.agent_code,
            "trace_type": self.trace_type,
            "model": actual_model or self.model_id,
            "status": status,
            "error_type": error_type,
            "duration_ms": int((time.perf_counter() - self._started) * 1000),
            "first_token_ms": self.first_token_ms,
            "llm_call_count": self.llm_call_count,
            "total_tokens": prompt + completion,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "step_count": step_count,
            "context_snapshot": {"items": self.context_items, "events": self.context_events},
            "error_detail": error_detail,
        }
        async with get_db_session() as db:
            await ai_trace_repository.insert_idempotent(db, values)

    def record_event(self, **kwargs) -> None:
        """记录推理期事件（护栏命中/计划快照/中断决策），settle 前调用写入 context_events"""
        if not self._settled:
            self.context_events.append(kwargs)

    @property
    def settled(self) -> bool:
        return self._settled


class LlmCallRecord:
    """单次 LLM 调用的采集记录（begin → observe → finish 三段式）"""

    def __init__(
        self,
        collector: TraceCollector,
        *,
        seq: int,
        step_position: int | None,
        model_id: str,
        messages: list[dict],
        system_prompt: str | None,
        tools: list[dict] | None,
    ) -> None:
        self._collector = collector
        self._seq = seq
        self._step_position = step_position
        self._model_id = model_id
        self._messages = messages
        self._system_prompt = system_prompt
        self._tools = tools
        self._started = time.perf_counter()
        self._first_token_ms: int | None = None
        self._text_parts: list[str] = []
        self._tool_calls: list[dict] = []
        self._usage: dict = {}
        # 候选路由 + 逐 Key 重试的物理调用尝试明细（B1 审计还原）
        self._attempts: list[dict] = []
        self._finished = False

    def observe_attempt(
        self,
        *,
        provider_id: int | None,
        key_id: int | None,
        model: str | None,
        status: int,
        error_code: str | None,
        latency_ms: int | None,
    ) -> None:
        """记录一次物理调用尝试（逐 Key/逐路由；status: 1成功/2失败/3跳过(熔断)）"""
        self._attempts.append(
            {
                "provider_id": provider_id,
                "key_id": key_id,
                "model": model,
                "status": status,
                "error_code": error_code,
                "latency_ms": latency_ms,
            }
        )

    def observe_chunk(self, chunk) -> None:
        """流式块观测（同步零开销路径，仅内存累积）"""
        if chunk.type in ("text_delta", "thinking_delta"):
            if self._first_token_ms is None:
                self._first_token_ms = int((time.perf_counter() - self._started) * 1000)
            if chunk.type == "text_delta":
                self._text_parts.append(chunk.content)
        elif chunk.type == "tool_call_complete":
            self._tool_calls.append(
                {
                    "name": chunk.tool_call_name,
                    "arguments": (chunk.content or "")[:_TOOL_ARGS_MAX_CHARS],
                }
            )
        elif chunk.type == "done" and chunk.usage:
            self._usage = chunk.usage

    async def finish(self, *, completed: bool, error_type: str | None = None) -> None:
        """调用结束落盘 sys_ai_llm_call（旁路：失败仅告警）"""
        if self._finished:
            return
        self._finished = True
        usage = self._usage
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        cached = usage.get("cached_tokens") or usage.get("cached_input_tokens") or 0
        status = CALL_STATUS_SUCCESS
        if not completed:
            status = CALL_STATUS_TIMEOUT if error_type == "timeout" else CALL_STATUS_FAILED
        values = {
            "trace_id": self._collector.trace_id,
            "seq": self._seq,
            "step_position": self._step_position,
            "model": self._model_id,
            "status": status,
            "error_type": error_type,
            "duration_ms": int((time.perf_counter() - self._started) * 1000),
            "first_token_ms": self._first_token_ms,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "tool_call": (
                {"has_tool_call": True, "tools": self._tool_calls} if self._tool_calls else None
            ),
            "input_snapshot": self._build_input_snapshot(),
            "output_snapshot": {
                "text": "".join(self._text_parts)[:OUTPUT_SNAPSHOT_MAX_CHARS],
                "tool_calls": self._tool_calls or None,
            },
            "attempts": self._attempts or None,
        }
        self._collector._aggregate_call(prompt, completion, cached, self._first_token_ms)
        try:
            async with get_db_session() as db:
                await ai_llm_call_repository.insert_idempotent(db, values)
        except Exception:
            logger.warning(
                "LLM 调用明细采集失败 trace_id=%s seq=%s", self._collector.trace_id, self._seq,
                exc_info=True,
            )

    def _build_input_snapshot(self) -> dict:
        """本轮输入构成（§2.3）：system 段/消息按角色计数/tools 定义/用户信息"""
        counts = {"user": 0, "assistant": 0, "tool": 0, "system": 0}
        messages_tokens = 0
        for msg in self._messages:
            role = msg.get("role")
            if role in counts:
                counts[role] += 1
            messages_tokens += _estimate_tokens(msg.get("content"))
        snapshot: dict = {
            "messages": {
                "counts": counts,
                "tokens": messages_tokens,
                "items": [
                    {
                        "role": msg.get("role") or "unknown",
                        "content": (msg.get("content") or "")[:MESSAGE_CONTENT_MAX_CHARS],
                    }
                    for msg in self._messages
                ],
            }
        }
        if self._system_prompt:
            snapshot["system_tokens"] = _estimate_tokens(self._system_prompt)
            snapshot["system_content"] = self._system_prompt[:SYSTEM_PROMPT_MAX_CHARS]
        if self._tools:
            snapshot["tool_count"] = len(self._tools)
            snapshot["tools"] = [
                {
                    "name": str(t.get("name", "")),
                    "description": (t.get("description") or "")[:TOOL_DESC_MAX_CHARS],
                }
                for t in self._tools
            ]
        if self._collector.user_id is not None:
            snapshot["user_id"] = self._collector.user_id
        return snapshot


_current_collector: ContextVar[TraceCollector | None] = ContextVar(
    "trace_collector", default=None
)


def start(
    *,
    conversation_id: int,
    message_id: int | None,
    user_id: int | None,
    agent_code: str | None,
    model_id: str | None,
) -> TraceCollector:
    """推理入口开启采集器（trace_id 复用日志链路，无请求上下文时降级生成）"""
    trace_id = _get_trace_id() or uuid4().hex
    collector = TraceCollector(
        trace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        agent_code=agent_code,
        model_id=model_id,
    )
    _current_collector.set(collector)
    return collector


def current() -> TraceCollector | None:
    return _current_collector.get()


def set_step_position(step: int) -> None:
    """同步当前推理步骤序号（awrap_model_call 步数递增后调用，供 step_position 关联）"""
    collector = _current_collector.get()
    if collector is not None:
        collector._step_position = step


def begin_llm_call(
    model_id: str,
    messages: list[dict],
    system_prompt: str | None,
    tools: list[dict] | None,
) -> LlmCallRecord | None:
    """llm_client 包装层入口：无采集器或已结算（旁路 LLM 调用）时返回 None 跳过采集"""
    collector = _current_collector.get()
    if collector is None or collector.settled:
        return None
    return collector.begin_llm_call(model_id, messages, system_prompt, tools)


async def finalize_success(
    *, usage: dict | None = None, step_count: int = 0, actual_model: str | None = None
) -> None:
    """推理正常结束落盘（after_agent 钩子外的收尾路径，如 direct 范式）"""
    collector = _current_collector.get()
    if collector is None:
        return
    try:
        await collector.settle(
            status=TRACE_STATUS_SUCCESS, usage=usage, step_count=step_count,
            actual_model=actual_model,
        )
    except Exception:
        logger.warning("过程链记录写入失败 trace_id=%s", collector.trace_id, exc_info=True)


async def finalize_unsettled(
    *, status: int, error_type: str | None = None, error_detail: Any | None = None
) -> None:
    """推理失败/中断/配额拒绝的收尾落盘（成功/失败/中断均写，旁路不抛错）"""
    collector = _current_collector.get()
    if collector is None:
        return
    try:
        await collector.settle(
            status=status, error_type=error_type, error_detail=error_detail
        )
    except Exception:
        logger.warning("过程链记录写入失败 trace_id=%s", collector.trace_id, exc_info=True)


@asynccontextmanager
async def bypass_span(
    *,
    conversation_id: int,
    message_id: int | None,
    user_id: int | None,
    model_id: str | None,
    trace_type: str,
) -> AsyncIterator[None]:
    """旁路 LLM 调用（摘要/记忆提取/建议/步骤摘要）独立过程链采集。

    with 块内的 LLM 调用经 begin_llm_call 感知本采集器（覆盖主链路采集器
    未开启/已结算时旁路调用无明细的盲区），退出时自动结算独立 trace
    （message_id 可为空，与主对话过程链经 trace_type 区分）。

    异常时落失败态后原样抛出——旁路函数自身的兜底逻辑仍生效，
    不改变原业务行为；退出后恢复外层采集器上下文。
    """
    prev_token = _current_collector.set(None)
    collector = start(
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        agent_code=None,
        model_id=model_id,
    )
    collector.trace_type = trace_type
    try:
        yield
    except Exception as e:
        await finalize_unsettled(status=TRACE_STATUS_FAILED, error_type=error_type_of(e))
        raise
    else:
        await finalize_success()
    finally:
        _current_collector.reset(prev_token)
