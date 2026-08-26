"""SseEventConverter：deepagents 流式事件（v2 格式）→ dehaze SSE

设计文档 §3.4 SSE 事件转换。deepagents 经 graph.astream(version="v2",
stream_mode=["messages","updates","custom"]) 产出 {type, ns, data} 事件，
本转换器逐条转换为 dehaze 自定义 SSE 事件并经 sse_emitter_manager 推送：

- messages + 文本增量 → content_block.start（首个 delta 前，type=text）→
  content_block.delta（text_delta）
- messages + 工具调用增量 → content_block.delta（input_json_delta）
- updates + 工具节点完成 → thought（推理步骤完成，落库 sys_ai_agent_thought）
- custom → custom
- updates 中的 __interrupt__ → interrupt

内容块收尾 content_block.stop 由推理层在 message.end 前调用 finish() 推送。

ns 标识事件来源（空为主 Agent，非空为子 Agent），子 Agent 的 thought
来源标记到 ns，计费经 middleware 归集主会话。
"""

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository

logger = logging.getLogger(__name__)


class SseEventConverter:
    """将 deepagents v2 流式事件转换为 dehaze SSE 事件。

    Args:
        ctx: 共享运行时上下文（stream_session_id/conversation_id/message_id 等）。
    """

    # 思考内容块独立索引（文本块固定 index=0）
    _THINKING_INDEX = 1

    def __init__(self, ctx: dict[str, Any]) -> None:
        self.ctx = ctx
        self._thought_position = 0
        # 文本内容块是否已推送 content_block.start（首个 text delta 前触发一次）
        self._text_block_started = False
        # 思考内容块状态：是否已推送 start / 当前是否打开 / 本轮思考文本累积（用于落库）
        self._thinking_block_started = False
        self._thinking_block_open = False
        self._thinking_buffer: list[str] = []
        # 工具调用跟踪：tool_call_id -> {name, args(增量拼接), started_at}
        # ToolMessage 返回时按 id 匹配，还原真实入参与耗时
        self._tool_calls: dict[str, dict[str, Any]] = {}

    async def _emit(self, event_type: str, data: dict) -> None:
        stream_session_id = self.ctx.get("stream_session_id")
        if stream_session_id:
            await sse_emitter_manager.send_event(stream_session_id, event_type, data)

    async def _ensure_text_block_start(self) -> None:
        """首个文本 delta 前推送 content_block.start（text），仅触发一次（§2.2 契约）。"""
        if self._text_block_started:
            return
        self._text_block_started = True
        await self._emit("content_block.start", {"index": 0, "type": "text"})

    async def finish(self) -> None:
        """流结束收尾：关闭打开的思考/文本内容块并落库残余思考（message.end 前）。"""
        if self._thinking_block_open:
            await self._emit("content_block.stop", {"index": self._THINKING_INDEX})
            self._thinking_block_open = False
        if self._thinking_buffer:
            await self._flush_thinking()
        if self._text_block_started:
            await self._emit("content_block.stop", {"index": 0})

    async def _open_thinking_block(self) -> None:
        """推送思考内容块 start（type=thinking，独立 index），仅触发一次。"""
        if self._thinking_block_started:
            return
        self._thinking_block_started = True
        self._thinking_block_open = True
        await self._emit(
            "content_block.start",
            {
                "index": self._THINKING_INDEX,
                "type": "thinking",
            },
        )

    async def _close_thinking_block(self) -> None:
        """关闭思考内容块（文本或工具调用开始时，思考段结束）。"""
        if not self._thinking_block_open:
            return
        await self._emit("content_block.stop", {"index": self._THINKING_INDEX})
        self._thinking_block_open = False

    async def _flush_thinking(self) -> None:
        """把本轮思考文本落为一条 agent_thought（thought=思考全文, tool=NULL）。"""
        if not self._thinking_buffer:
            return
        text = "".join(self._thinking_buffer).strip()
        self._thinking_buffer = []
        if not text:
            return
        await self._record_thought(
            tool=None,
            tool_input=None,
            observation=None,
            thought=text,
            status=1,
            error=None,
            latency_ms=0,
        )

    async def _record_thought(
        self,
        tool: str | None,
        tool_input: Any,
        observation: str | None,
        *,
        thought: str,
        status: int,
        error: str | None,
        latency_ms: int,
    ) -> None:
        """落库推理步骤并推送 thought 事件（思考/工具步骤共用）。

        status: 1 成功 / 2 失败 / 3 跳过；error 为失败原因（status=2 时透出）。
        """
        self._thought_position += 1
        try:
            async with get_db_session() as db:
                await ai_agent_thought_repository.create_thought(
                    db,
                    message_id=self.ctx.get("message_id"),
                    conversation_id=self.ctx.get("conversation_id"),
                    position=self._thought_position,
                    thought=thought,
                    tool=tool,
                    tool_input=tool_input,
                    observation=observation,
                    status=status,
                    error=error,
                    latency_ms=latency_ms,
                )
        except Exception:
            logger.warning("thought 落库失败", exc_info=True)
        await self._emit(
            "thought",
            {
                "position": self._thought_position,
                "thought": thought,
                "tool": tool,
                "toolInput": tool_input,
                "observation": observation,
                "status": status,
                "error": error,
                "latencyMs": latency_ms,
            },
        )

    async def record_thought(
        self,
        tool: str,
        observation: str,
        *,
        thought: str | None = None,
        tool_input: Any = None,
        position: int | None = None,
        status: int = 1,
    ) -> None:
        """公开的推理步骤记录接口（供多步推理范式 middleware 经 custom 通道调用）。

        position 缺省自增；同批并行子任务可显式指定相同 position 以标注批次归属，
        并行批 metadata 经 observation 携带 batch 标注。
        """
        if position is not None:
            self._thought_position = max(self._thought_position, position)
        await self._record_thought(
            tool=tool,
            tool_input=tool_input,
            observation=observation,
            thought=thought or f"调用工具: {tool}",
            status=status,
            error=None,
            latency_ms=0,
        )

    async def record_plan(self, plan: dict[str, Any], phase: str) -> None:
        """推送 plan SSE 事件（§2.2 契约：{tasks, status, revisions}）。"""
        tasks = [
            {
                "id": t.get("id"),
                "description": t.get("description"),
                "dependsOn": t.get("depends_on") or [],
                "status": t.get("status", "pending"),
            }
            for t in (plan.get("tasks") or [])
        ]
        await self._emit(
            "plan",
            {
                "tasks": tasks,
                "status": plan.get("status", "pending"),
                "revisions": plan.get("revisions") or [],
                "phase": phase,
            },
        )

    async def handle(self, event: dict[str, Any]) -> None:
        """处理单条 deepagents v2 事件，推送对应的 dehaze SSE 事件。"""
        etype = event.get("type")
        data = event.get("data")

        if etype == "messages":
            await self._handle_messages(data)
        elif etype == "updates":
            await self._handle_updates(data)
        elif etype == "custom":
            await self._handle_custom(data)
        else:
            # 其他类型（values 等）忽略，避免前端收到意外事件
            return

    async def _handle_custom(self, data: Any) -> None:
        """custom 事件：识别多步推理范式推送的 plan/thought 结构化事件，否则原样透传。

        范式 middleware 经图 custom 通道推送 {type, data} 载荷：
        - type=plan：推送 plan SSE
        - type=thought：经公开接口落库并推送 thought
        """
        if isinstance(data, dict) and data.get("type") == "plan":
            payload = data.get("data") or {}
            await self.record_plan(payload.get("plan") or {}, payload.get("phase", "plan"))
            return
        if isinstance(data, dict) and data.get("type") == "thought":
            payload = data.get("data") or {}
            await self.record_thought(
                tool=payload.get("tool", ""),
                observation=payload.get("observation", ""),
                thought=payload.get("thought"),
                tool_input=payload.get("tool_input"),
                position=payload.get("position"),
            )
            return
        await self._emit("custom", {"data": data})

    async def _handle_messages(self, data: Any) -> None:
        """messages 事件：data 形如 [AIMessageChunk, metadata]，提取文本/工具调用增量。

        文本内容块在首个 text delta 前推送 content_block.start；工具调用增量同时
        累积到 _tool_calls（首个 chunk 携带 id/name，后续 chunk 仅携带 args 增量），
        ToolMessage 返回时按 tool_call_id 匹配还原完整入参。
        """
        if not isinstance(data, (list, tuple)) or not data:
            return
        chunk = data[0]
        if not isinstance(chunk, AIMessageChunk):
            return
        # 思考流：经 DehazeChatModel 透传的 thinking 增量独立成块推送并累积（tool=NULL）
        thinking = chunk.additional_kwargs.get("thinking")
        if thinking:
            await self._open_thinking_block()
            self._thinking_buffer.append(thinking)
            await self._emit(
                "content_block.delta",
                {
                    "index": self._THINKING_INDEX,
                    "delta": {"type": "thinking_delta", "thinking": thinking},
                },
            )
        if chunk.content:
            # 文本段开始时关闭思考块（思考在前、回复在后，按模型真实输出序不强行重排）
            await self._close_thinking_block()
            await self._ensure_text_block_start()
            await self._emit(
                "content_block.delta",
                {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk.content},
                },
            )
        latest_tc_id = None
        if chunk.tool_call_chunks:
            # 工具调用开始时关闭思考块（思考→行动切换，思考段结束）
            await self._close_thinking_block()
        for tc in chunk.tool_call_chunks:
            tc_id = tc.get("id") or latest_tc_id
            if tc_id:
                latest_tc_id = tc_id
                entry = self._tool_calls.setdefault(
                    tc_id, {"name": "", "args": "", "started_at": time.monotonic()}
                )
                if tc.get("name"):
                    entry["name"] = tc["name"]
                if tc.get("args"):
                    entry["args"] += tc["args"]
            if tc.get("args"):
                await self._emit(
                    "content_block.delta",
                    {
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc["args"],
                            "name": tc.get("name", ""),
                        },
                    },
                )

    async def _handle_updates(self, data: Any) -> None:
        """updates 事件：data 为 {node_name: channel_updates}。

        工具节点完成 → 按 tool_call_id 匹配累积的调用信息记录 thought（含真实入参与耗时）；
        含 __interrupt__ → 推送 interrupt。
        """
        if not isinstance(data, dict):
            return
        for node, updates in data.items():
            if node == "__interrupt__":
                # 统一推送 interrupt SSE（§3.4 契约 {type, data}）。中断数据源
                # （如 algorithm_recommend_service.interrupt_data）value 内含 type
                # 字段，提升到事件顶层对齐契约；data 取 value.data 业务载荷。
                for item in updates or []:
                    value = item.value if isinstance(item.value, dict) else {}
                    await self._emit(
                        "interrupt",
                        {
                            "type": value.get("type", "confirm"),
                            "data": value.get("data", value),
                        },
                    )
                continue
            messages = updates.get("messages") if isinstance(updates, dict) else None
            if not messages:
                continue
            # 工具节点返回的 ToolMessage 记录为 thought（先落本条思考，再落工具步骤并列）
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    await self._close_thinking_block()
                    await self._flush_thinking()
                    entry = self._tool_calls.pop(msg.tool_call_id, None)
                    latency_ms = (
                        int((time.monotonic() - entry["started_at"]) * 1000) if entry else 0
                    )
                    # 错误恢复状态由 DehazeHooksMiddleware 写入 additional_kwargs，
                    # 默认成功（status=1）
                    recovery = msg.additional_kwargs or {}
                    status = int(recovery.get("_dehaze_status", 1))
                    error = recovery.get("_dehaze_error")
                    await self._record_thought(
                        tool=msg.name or (entry or {}).get("name") or "",
                        tool_input=_parse_tool_args(entry) if entry else msg.tool_call_id,
                        observation=str(msg.content)[:500],
                        thought=f"调用工具: {msg.name or (entry or {}).get('name') or ''}",
                        status=status,
                        error=error,
                        latency_ms=latency_ms,
                    )


def _parse_tool_args(entry: dict[str, Any]) -> Any:
    """将累积的工具调用参数 JSON 文本解析为 dict，解析失败保留原始文本。"""
    try:
        return json.loads(entry["args"]) if entry["args"] else {}
    except ValueError:
        return entry["args"]
