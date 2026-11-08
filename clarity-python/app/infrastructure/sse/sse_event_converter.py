"""SseEventConverter：deepagents 流式事件（v2 格式）→ clarity SSE

deepagents 经 graph.astream(version="v2", stream_mode=["messages","updates","custom"])
产出 {type, ns, data} 事件，本转换器逐条转换为 clarity 的块事件并经
sse_emitter_manager 推送。

事件白名单（契约 Event 定义，顺序即生命周期）：
message.start → block.start → block.delta → block.stop → plan.update / interrupt /
usage / suggestions → message.end；异常另有 error。

块模型：一条消息的内容只有 Block[]，kind 取 text/thinking/tool_use。
- 同一时刻只有一个块处于打开态，切换 kind（思考→正文→工具）即关闭前一块；
- seq 表达"同一条消息内同 kind 块的先后"，按 kind 各自从 1 递增（resume 时从
  DB 现有最大值续号），渲染按分区保序，不做跨 kind 交错；
- tool_use 每次调用一个独立块，工具返回时用真实入参/输出/耗时收尾。

转换器同时持有本轮终态快照（blocks/plan/pendingInterrupt），由推理层在
finalize 时整体落库，保证刷新后读到的形状与流式终态一致。
"""

import copy
import json
import logging
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.models.entity.sys_ai_message import (
    BLOCK_KIND_TEXT,
    BLOCK_KIND_THINKING,
    BLOCK_KIND_TOOL_USE,
)
from app.repository.ai_message_repository import ai_message_repository
from app.service.ai.service import trace_collector
from app.service.ai.service.contract_payload import build_pending_interrupt, to_contract_plan
from app.utils.pii import StreamingPiiMasker

logger = logging.getLogger(__name__)

# 工具执行状态（ToolStatus）
_TOOL_STATUS = {1: "success", 2: "failed", 3: "skipped"}


def _snapshot(block: dict[str, Any]) -> dict[str, Any]:
    """块快照：事件载荷与内存态解耦，避免后续增量污染已下发的事件。"""
    return copy.deepcopy(block)


class SseEventConverter:
    """将 deepagents v2 流式事件转换为 clarity SSE 事件。

    Args:
        ctx: 共享运行时上下文（stream_session_id/conversation_id/message_id/seq_base）。
    """

    def __init__(self, ctx: dict[str, Any]) -> None:
        self.ctx = ctx
        self.message_id: int = int(ctx.get("message_id") or 0)
        # 终态快照：按块的创建顺序保存（读模型按 kind + seq 分区渲染）；
        # resume 续流时以 DB 已落库的块为基底，只追加本轮新块
        self.blocks: list[dict[str, Any]] = list(ctx.get("blocks") or [])
        # 各 kind 的 seq 基线：续流（resume）时由 DB 现有最大 seq 续号
        self._seq: dict[str, int] = dict(ctx.get("seq_base") or {})
        self._open_block: dict[str, Any] | None = None
        self._open_started_at: float = 0.0
        # 工具调用跟踪：call_id -> {block, args, started_at}
        self._tool_calls: dict[str, dict[str, Any]] = {}
        self.plan: dict[str, Any] | None = None
        self.pending_interrupt: dict[str, Any] | None = None
        # 当前事件来源 Agent 归属（handle 逐事件解析 ns 更新）
        self._agent_code: str | None = None
        self._is_subagent = False
        # 流式出口脱敏：PII/凭据逐块外发前脱敏，跨 chunk 的敏感串由 masker 兜住
        self._text_masker = StreamingPiiMasker()
        self._thinking_masker = StreamingPiiMasker()

    # ── 事件出口 ────────────────────────────────────────

    async def _emit(self, event_type: str, data: dict) -> None:
        stream_session_id = self.ctx.get("stream_session_id")
        if stream_session_id:
            await sse_emitter_manager.send_event(stream_session_id, event_type, data)

    def blocks_snapshot(self) -> list[dict[str, Any]]:
        """本轮块快照（finalize 整体落库用）。"""
        return copy.deepcopy(self.blocks)

    def _next_seq(self, kind: str) -> int:
        self._seq[kind] = self._seq.get(kind, 0) + 1
        return self._seq[kind]

    # ── 块生命周期 ──────────────────────────────────────

    async def _open(self, kind: str, *, tool: dict[str, Any] | None = None) -> dict[str, Any]:
        await self._close_open_block()
        block: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "seq": self._next_seq(kind),
            "durationMs": None,
        }
        if kind == BLOCK_KIND_TOOL_USE:
            block["tool"] = tool
        else:
            block["text"] = ""
        self.blocks.append(block)
        self._open_block = block
        self._open_started_at = time.monotonic()
        await self._emit("block.start", {"messageId": self.message_id, "block": _snapshot(block)})
        return block

    async def _close_open_block(self) -> None:
        block = self._open_block
        if block is None:
            return
        # 关闭块前先冲刷该通道的脱敏暂存：暂存尾部属于本块内容，若留到下一个块
        # 再 flush，会整体迁到后续块（跨块丢字/错位）；flush 同时重置 masker
        if block["kind"] == BLOCK_KIND_THINKING:
            await self._flush_thinking_tail()
        elif block["kind"] == BLOCK_KIND_TEXT:
            await self._flush_text_tail()
        self._open_block = None
        block["durationMs"] = max(int((time.monotonic() - self._open_started_at) * 1000), 0)
        await self._emit("block.stop", {"messageId": self.message_id, "block": _snapshot(block)})

    async def _delta(self, block: dict[str, Any], delta: str) -> None:
        if not delta:
            return
        if block["kind"] != BLOCK_KIND_TOOL_USE:
            block["text"] = (block.get("text") or "") + delta
        await self._emit(
            "block.delta",
            {"messageId": self.message_id, "blockId": block["id"], "delta": delta},
        )

    async def _ensure_block(self, kind: str) -> dict[str, Any]:
        block = self._open_block
        if block is not None and block["kind"] == kind:
            return block
        return await self._open(kind)

    async def finish(self) -> None:
        """流结束收尾：冲刷脱敏残留、关闭仍打开的块（message.end 之前）。

        _close_open_block 已按块 kind 冲刷对应 masker，此处无需重复 flush。
        """
        await self._close_open_block()

    async def flush_final_text(self, fallback: str) -> list[dict[str, Any]]:
        """终态兜底：整轮没有任何正文块、但图 state 产出了最终答复时补一个 text 块。

        计划执行等范式在 state 里组装答复、未逐字流式下发，若不补块会出现
        "completed + 空 blocks"，刷新后正文凭空消失。补块同时下发块事件，
        在线客户端与刷新后读到的内容一致。
        """
        has_text = any(b.get("kind") == BLOCK_KIND_TEXT and b.get("text") for b in self.blocks)
        if fallback and not has_text:
            block = await self._open(BLOCK_KIND_TEXT)
            await self._delta(block, fallback)
            await self._close_open_block()
        return self.blocks_snapshot()

    async def _flush_text_tail(self) -> None:
        block = self._open_block
        pending = self._text_masker.flush()
        if block is not None and pending:
            await self._delta(block, pending)

    async def _flush_thinking_tail(self) -> None:
        block = self._open_block
        pending = self._thinking_masker.flush()
        if block is not None and pending:
            await self._delta(block, pending)

    # ── 计划 / 中断 ─────────────────────────────────────

    async def record_plan(self, plan: dict[str, Any], phase: str) -> None:
        """plan.update 到达即持久化到 message.plan，并推送契约 Plan。

        计划是消息级事实源：刷新后从 message 读回同一份计划（含修订记录 actor/at），
        不依赖流式期间的内存副本。
        """
        payload = to_contract_plan(plan, phase=phase)
        self.plan = payload
        if self.message_id:
            try:
                async with get_db_session() as db:
                    await ai_message_repository.update_plan(db, self.message_id, payload)
            except Exception:
                logger.warning("计划持久化失败: msg=%s", self.message_id, exc_info=True)
        await self._emit("plan.update", {"messageId": self.message_id, "plan": payload})
        collector = trace_collector.current()
        if collector is not None:
            try:
                collector.record_event(
                    event="plan",
                    phase=phase,
                    plan_summary=json.dumps(plan, ensure_ascii=False)[:2000],
                )
            except Exception:
                logger.warning("plan 过程链事件记录失败", exc_info=True)

    async def _handle_interrupt(self, items: Any) -> None:
        """__interrupt__ → 契约 Interrupt：落库 pending_interrupt 后下发。"""
        for item in items or []:
            value = item.value if isinstance(item.value, dict) else {}
            itype = str(value.get("type") or "confirm")
            payload = build_pending_interrupt({"type": itype, "data": value.get("data", value)})
            if payload is None:
                continue
            self.pending_interrupt = payload
            if self.message_id:
                try:
                    async with get_db_session() as db:
                        await ai_message_repository.update_pending_interrupt(
                            db, self.message_id, payload
                        )
                except Exception:
                    logger.warning(
                        "中断落库失败: msg=%s", self.message_id, exc_info=True
                    )
            await self._emit(
                "interrupt", {"messageId": self.message_id, "interrupt": payload}
            )

    # ── 事件分发 ────────────────────────────────────────

    async def handle(self, event: dict[str, Any]) -> None:
        """处理单条 deepagents v2 事件，推送对应的 clarity SSE 事件。"""
        etype = event.get("type")
        data = event.get("data")
        # ns 非空即子 Agent/Team 子图事件（主图为空元组）；段格式"节点名:task_id"，
        # 嵌套时多段，取最后一段的节点名部分作为子 Agent 标识
        ns = event.get("ns")
        if ns:
            self._agent_code = str(ns[-1]).split(":")[0]
            self._is_subagent = True
        else:
            self._agent_code = None
            self._is_subagent = False

        if etype == "messages":
            await self._handle_messages(data)
        elif etype == "updates":
            await self._handle_updates(data)
        elif etype == "custom":
            await self._handle_custom(data)
        # 其他类型（values 等）忽略，避免前端收到契约外事件

    async def _handle_custom(self, data: Any) -> None:
        """custom 事件：仅识别多步推理范式的 plan 结构化事件，其余忽略。"""
        if isinstance(data, dict) and data.get("type") == "plan":
            payload = data.get("data") or {}
            await self.record_plan(payload.get("plan") or {}, payload.get("phase", "plan"))

    async def _handle_messages(self, data: Any) -> None:
        """messages 事件：data 形如 [AIMessageChunk, metadata]。

        thinking 增量进 thinking 块，正文增量进 text 块；工具调用首见即开一个独立的
        tool_use 块并累积参数增量，ToolMessage 返回时按 tool_call_id 收尾。
        """
        if not isinstance(data, (list, tuple)) or not data:
            return
        chunk = data[0]
        if not isinstance(chunk, AIMessageChunk):
            return
        thinking = chunk.additional_kwargs.get("thinking")
        if thinking:
            block = await self._ensure_block(BLOCK_KIND_THINKING)
            await self._delta(block, self._thinking_masker.push(thinking))
        if chunk.content:
            # 思考在前、回复在后：正文开始即关闭思考块
            if self._open_block is not None and self._open_block["kind"] == BLOCK_KIND_THINKING:
                await self._flush_thinking_tail()
                await self._close_open_block()
            block = await self._ensure_block(BLOCK_KIND_TEXT)
            await self._delta(block, self._text_masker.push(chunk.text))
        latest_tc_id = None
        for tc in chunk.tool_call_chunks:
            tc_id = tc.get("id") or latest_tc_id
            if not tc_id:
                continue
            latest_tc_id = tc_id
            entry = self._tool_calls.get(tc_id)
            if entry is None:
                tool = {
                    "callId": tc_id,
                    "name": tc.get("name") or "",
                    "input": None,
                    "output": None,
                    "summary": None,
                    "status": "running",
                    "error": None,
                    "latencyMs": None,
                    "agentCode": self._agent_code,
                    "subAgent": self._is_subagent,
                }
                block = await self._open(BLOCK_KIND_TOOL_USE, tool=tool)
                entry = {
                    "block": block,
                    "args": "",
                    "started_at": time.monotonic(),
                }
                self._tool_calls[tc_id] = entry
            elif tc.get("name") and not entry["block"]["tool"]["name"]:
                entry["block"]["tool"]["name"] = tc["name"]
            if tc.get("args"):
                entry["args"] += tc["args"]
                # 工具入参以增量形式外发，前端按 blockId 累积参数草稿
                await self._delta(entry["block"], tc["args"])

    async def _handle_updates(self, data: Any) -> None:
        """updates 事件：工具节点完成 → 收尾对应 tool_use 块；__interrupt__ → 中断。"""
        if not isinstance(data, dict):
            return
        for node, updates in data.items():
            if node == "__interrupt__":
                await self._handle_interrupt(updates)
                continue
            messages = updates.get("messages") if isinstance(updates, dict) else None
            if not messages:
                continue
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    await self._complete_tool(msg)

    async def _complete_tool(self, msg: ToolMessage) -> None:
        """工具返回：用真实入参、输出、耗时与执行状态收尾该 tool_use 块。"""
        entry = self._tool_calls.pop(msg.tool_call_id, None)
        block = (entry or {}).get("block")
        if block is None:
            return
        tool = block["tool"]
        if msg.name and not tool["name"]:
            tool["name"] = msg.name
        tool["input"] = _parse_tool_args(entry)
        tool["output"] = str(msg.content)
        tool["latencyMs"] = max(int((time.monotonic() - entry["started_at"]) * 1000), 0)
        # 错误恢复状态由 DehazeHooksMiddleware 写入 additional_kwargs，默认成功
        recovery = msg.additional_kwargs or {}
        status = int(recovery.get("_dehaze_status", 1))
        tool["status"] = _TOOL_STATUS.get(status, "success")
        tool["error"] = recovery.get("_dehaze_error")
        summary = recovery.get("_dehaze_summary")
        tool["summary"] = summary if summary is not None else None
        if self._open_block is block:
            self._open_block = None
            block["durationMs"] = tool["latencyMs"]
        await self._emit("block.stop", {"messageId": self.message_id, "block": _snapshot(block)})


def _parse_tool_args(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """将累积的工具调用参数 JSON 文本解析为 dict；解析失败时保留原文放在入参外层。"""
    if entry is None:
        return None
    raw = entry["args"]
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}
