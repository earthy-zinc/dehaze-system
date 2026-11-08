"""SseEventConverter：deepagents 事件 → 契约块事件。

只允许 message.start / block.start|delta|stop / plan.update / interrupt / usage /
suggestions / message.end / error 八类事件；旧的 content_block.*、thought、custom
一律不得出现。
"""

import re

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.infrastructure.sse.sse_event_converter import SseEventConverter
from app.service.ai.service import trace_collector
from tests.stubs.fakes import RecorderEmitter

_HEX32 = re.compile(r"^[0-9a-f]{32}$")

_FORBIDDEN = (
    "content_block.start",
    "content_block.delta",
    "content_block.stop",
    "thought",
    "custom",
)


class _Repo:
    def __init__(self):
        self.plans = []
        self.interrupts = []

    async def update_plan(self, db, msg_id, plan):
        self.plans.append((msg_id, plan))

    async def update_pending_interrupt(self, db, msg_id, interrupt):
        self.interrupts.append((msg_id, interrupt))


class _Session:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def conv(monkeypatch):
    ctx = {"stream_session_id": "s1", "message_id": 1, "conversation_id": 100}
    emitter = RecorderEmitter()
    repo = _Repo()
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter)
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.ai_message_repository", repo)
    monkeypatch.setattr(
        "app.infrastructure.sse.sse_event_converter.get_db_session", lambda: _Session()
    )
    return SseEventConverter(ctx), emitter, repo


def _types(emitter) -> list[str]:
    return [e[0] for e in emitter.events]


def _texts(emitter) -> str:
    return "".join(
        e[1]["delta"]
        for e in emitter.events
        if e[0] == "block.delta" and e[1]["blockId"] == _text_block_id(emitter)
    )


def _text_block_id(emitter) -> str:
    return next(
        (
            e[1]["block"]["id"]
            for e in emitter.events
            if e[0] == "block.start" and e[1]["block"]["kind"] == "text"
        ),
        "",
    )


def _block_deltas(emitter, kind: str) -> str:
    """按块 kind 归集增量（块类型从 block.start 反查）。"""
    kinds = {
        e[1]["block"]["id"]: e[1]["block"]["kind"] for e in emitter.events if e[0] == "block.start"
    }
    return "".join(
        e[1]["delta"]
        for e in emitter.events
        if e[0] == "block.delta" and kinds.get(e[1]["blockId"]) == kind
    )


class TestTextBlocks:
    async def test_block_start_then_delta_then_stop(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="你好")]}
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="世界")]}
        )
        await converter.finish()

        assert _types(emitter) == ["block.start", "block.delta", "block.delta", "block.stop"]
        start = emitter.events[0][1]
        assert start["messageId"] == 1
        assert start["block"]["kind"] == "text"
        assert start["block"]["seq"] == 1
        assert _HEX32.match(start["block"]["id"])
        assert [e[1]["delta"] for e in emitter.events if e[0] == "block.delta"] == ["你好", "世界"]
        stop = emitter.events[-1][1]["block"]
        assert stop["text"] == "你好世界"
        assert stop["durationMs"] >= 0

    async def test_no_text_no_block(self, conv):
        converter, emitter, _ = conv
        await converter.finish()
        assert _types(emitter) == []

    async def test_text_block_seq_increases_after_tool(self, conv):
        """同 kind 多块 seq 递增：正文→工具→正文，第二段正文 seq=2。"""
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="甲")]}
        )
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": "{}", "id": "c1", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="乙")]}
        )
        await converter.finish()

        text_starts = [
            e[1]["block"]
            for e in emitter.events
            if e[0] == "block.start" and e[1]["block"]["kind"] == "text"
        ]
        assert [b["seq"] for b in text_starts] == [1, 2]
        # 打开新块前先关上一块：块生命周期成对出现
        assert _types(emitter).count("block.start") == _types(emitter).count("block.stop")

    async def test_seq_base_continues_on_resume(self, monkeypatch):
        """续流：seq 从 DB 现有最大值续号，已有块作为快照基底保留。"""
        existing = [{"id": "old", "kind": "text", "seq": 3, "text": "旧正文", "durationMs": 5}]
        emitter = RecorderEmitter()
        monkeypatch.setattr(
            "app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter
        )
        converter = SseEventConverter(
            {
                "stream_session_id": "s2",
                "message_id": 1,
                "conversation_id": 100,
                "blocks": existing,
                "seq_base": {"text": 3},
            }
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="续")]}
        )
        await converter.finish()

        start = next(e[1]["block"] for e in emitter.events if e[0] == "block.start")
        assert start["seq"] == 4
        snapshot = converter.blocks_snapshot()
        assert [b["seq"] for b in snapshot] == [3, 4]
        assert snapshot[0]["text"] == "旧正文"


class TestThinkingBlocks:
    async def test_thinking_is_own_block(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "思考A"})],
            }
        )
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "思考B"})],
            }
        )
        await converter.finish()

        start = emitter.events[0][1]["block"]
        assert start["kind"] == "thinking"
        assert start["seq"] == 1
        assert _block_deltas(emitter, "thinking") == "思考A思考B"
        assert emitter.events[-1][1]["block"]["text"] == "思考A思考B"

    async def test_text_closes_thinking_block(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "先想"})],
            }
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="正文")]}
        )
        await converter.finish()

        stops = [e[1]["block"] for e in emitter.events if e[0] == "block.stop"]
        assert stops[0]["kind"] == "thinking"
        assert stops[0]["text"] == "先想"
        assert stops[1]["kind"] == "text"

    async def test_each_thinking_segment_own_block(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "思考1"})],
            }
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="回复1")]}
        )
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "思考2"})],
            }
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="回复2")]}
        )
        await converter.finish()

        thinking = [
            e[1]["block"]
            for e in emitter.events
            if e[0] == "block.start" and e[1]["block"]["kind"] == "thinking"
        ]
        assert [b["seq"] for b in thinking] == [1, 2]
        assert [b["seq"] for b in converter.blocks_snapshot() if b["kind"] == "thinking"] == [1, 2]


class TestToolBlocks:
    async def test_tool_call_opens_independent_block(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": '{"q":', "id": "c1", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})

        start = emitter.events[0][1]["block"]
        assert start["kind"] == "tool_use"
        assert start["seq"] == 1
        assert start["tool"]["callId"] == "c1"
        assert start["tool"]["name"] == "search"
        assert start["tool"]["status"] == "running"
        assert start["tool"]["subAgent"] is False
        assert emitter.events[1][0] == "block.delta"
        assert emitter.events[1][1]["delta"] == '{"q":'

    async def test_parallel_tool_calls_two_blocks(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {"name": "a", "args": "{}", "id": "c1", "index": 0},
            {"name": "b", "args": "{}", "id": "c2", "index": 1},
        ]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})

        tool_blocks = [e[1]["block"] for e in emitter.events if e[0] == "block.start"]
        assert [b["tool"]["callId"] for b in tool_blocks] == ["c1", "c2"]
        assert [b["seq"] for b in tool_blocks] == [1, 2]
        assert tool_blocks[0]["id"] != tool_blocks[1]["id"]

    async def test_tool_message_completes_block(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": '{"q": "雾"}', "id": "c1", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "tool_node": {
                        "messages": [ToolMessage(content="ok", tool_call_id="c1", name="search")]
                    }
                },
            }
        )

        stop = next(e[1]["block"] for e in emitter.events if e[0] == "block.stop")
        assert stop["tool"]["status"] == "success"
        assert stop["tool"]["input"] == {"q": "雾"}
        assert stop["tool"]["output"] == "ok"
        assert stop["tool"]["latencyMs"] >= 0

    async def test_tool_failure_status_mapped(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "dehaze", "args": "{}", "id": "c2", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "tool_node": {
                        "messages": [
                            ToolMessage(
                                content="超时",
                                tool_call_id="c2",
                                name="dehaze",
                                status="error",
                                additional_kwargs={"_dehaze_status": 2, "_dehaze_error": "超时"},
                            )
                        ]
                    }
                },
            }
        )
        tool = next(e[1]["block"] for e in emitter.events if e[0] == "block.stop")["tool"]
        assert tool["status"] == "failed"
        assert tool["error"] == "超时"

    async def test_subagent_tool_attribution(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": "{}", "id": "c3", "index": 0}]
        await converter.handle({"type": "messages", "ns": ["task:abc"], "data": [chunk]})
        tool = next(e[1]["block"] for e in emitter.events if e[0] == "block.start")["tool"]
        assert tool["agentCode"] == "task"
        assert tool["subAgent"] is True

    async def test_finish_closes_open_tool_block(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": "{", "id": "c9", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.finish()
        assert _types(emitter)[-1] == "block.stop"


class TestPlanAndInterrupt:
    async def test_plan_update_persisted_and_emitted(self, conv):
        converter, emitter, repo = conv
        await converter.record_plan(
            {
                "tasks": [
                    {"id": "A", "description": "分析", "depends_on": ["B"], "status": "done"},
                    {"id": "B", "description": "执行", "status": "pending"},
                ],
                "status": "executing",
                "revisions": [{"revisionNo": 1, "reason": "依赖调整"}],
            },
            phase="plan",
        )

        event = emitter.events[0]
        assert event[0] == "plan.update"
        plan = event[1]["plan"]
        assert event[1]["messageId"] == 1
        assert plan["status"] == "running"
        assert plan["phase"] == "plan"
        assert plan["awaitingApproval"] is False
        assert [t["status"] for t in plan["tasks"]] == ["completed", "pending"]
        assert plan["tasks"][0]["dependsOn"] == ["B"]
        # 修订记录补齐 actor 与 at
        assert plan["revisions"][0]["actor"] == "assistant"
        assert plan["revisions"][0]["at"]
        # 到达即落库
        assert repo.plans[0][0] == 1
        assert repo.plans[0][1]["status"] == "running"

    async def test_plan_pending_awaits_approval(self, conv):
        converter, emitter, _ = conv
        await converter.record_plan({"tasks": [], "status": "pending"}, phase="plan")
        assert emitter.events[0][1]["plan"]["awaitingApproval"] is True

    async def test_interrupt_is_contract_shaped_and_persisted(self, conv):
        from langgraph.types import Interrupt

        converter, emitter, repo = conv
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "type": "confirm",
                                "data": {
                                    "confirmKind": "algorithm_recommend",
                                    "recommendation": {"modelId": "m1", "modelName": "去雾A"},
                                },
                            }
                        )
                    ]
                },
            }
        )

        event = emitter.events[0]
        assert event[0] == "interrupt"
        interrupt = event[1]["interrupt"]
        assert interrupt["type"] == "confirm"
        assert interrupt["confirmKind"] == "algorithm_recommend"
        assert interrupt["recommendation"]["modelName"] == "去雾A"
        assert interrupt["actionable"] is True
        assert interrupt["createdAt"]
        assert interrupt["expiresAt"]
        assert interrupt["id"]
        assert repo.interrupts[0][1]["id"] == interrupt["id"]
        assert converter.pending_interrupt == interrupt

    async def test_plan_approve_carries_contract_tasks(self, conv):
        from langgraph.types import Interrupt

        converter, emitter, _ = conv
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "type": "plan_approve",
                                "data": {
                                    "plan": {
                                        "tasks": [
                                            {"id": "A", "description": "分析", "tool_hint": "x"}
                                        ],
                                        "status": "pending",
                                    }
                                },
                            }
                        )
                    ]
                },
            }
        )
        interrupt = emitter.events[0][1]["interrupt"]
        assert interrupt["type"] == "plan_approve"
        assert interrupt["plan"] == [
            {
                "id": "A",
                "description": "分析",
                "dependsOn": [],
                "status": "pending",
                "paradigm": None,
            }
        ]


class TestEventWhitelist:
    async def test_only_contract_events_emitted(self, conv):
        from langgraph.types import Interrupt

        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="正文")]}
        )
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="", additional_kwargs={"thinking": "想"})],
            }
        )
        await converter.handle({"type": "custom", "ns": [], "data": {"k": 1}})
        await converter.handle(
            {"type": "custom", "ns": [], "data": {"type": "thought", "data": {}}}
        )
        await converter.handle({"type": "values", "ns": [], "data": {}})
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {"__interrupt__": [Interrupt(value={"foo": "bar"})]},
            }
        )
        await converter.finish()

        types = set(_types(emitter))
        assert types <= {"block.start", "block.delta", "block.stop", "interrupt"}
        for forbidden in _FORBIDDEN:
            assert forbidden not in types

    async def test_no_stream_session_skips_emit(self, monkeypatch):
        converter = SseEventConverter({"stream_session_id": None})
        emitter = RecorderEmitter()
        monkeypatch.setattr(
            "app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="x")]}
        )
        assert emitter.events == []


class TestPiiMaskingAtStreamExit:
    """PII 脱敏必须在 SSE 推送前完成：事后脱敏等于已经把敏感信息推给了客户端。"""

    async def test_phone_split_across_chunks_masked(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="联系138")]}
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="12345678，谢谢")]}
        )
        await converter.finish()
        assert _texts(emitter) == "联系***，谢谢"

    async def test_secret_key_masked(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="key: sk-abcdef123456")]}
        )
        await converter.finish()
        assert _texts(emitter) == "key: ***"

    async def test_thinking_channel_masked(self, conv):
        converter, emitter, _ = conv
        chunk = AIMessageChunk(content="", additional_kwargs={"thinking": "手机号13812345678"})
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.finish()
        assert _block_deltas(emitter, "thinking") == "手机号***"

    async def test_plain_text_not_delayed(self, conv):
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="你好")]}
        )
        assert _texts(emitter) == "你好"

    async def test_pending_tail_flushed_before_block_switch(self, conv):
        """关闭 text 块前冲刷脱敏暂存：否则暂存尾部会迁移到下一个 text 块。"""
        converter, emitter, _ = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="电话138")]}
        )
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "search", "args": "{}", "id": "c1", "index": 0}]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="12345678，谢谢")]}
        )
        await converter.finish()

        text_stops = [
            e[1]["block"]
            for e in emitter.events
            if e[0] == "block.stop" and e[1]["block"]["kind"] == "text"
        ]
        assert [b["text"] for b in text_stops] == ["电话138", "12345678，谢谢"]

    async def test_plan_trace_event_recorded(self, conv):
        converter, _emitter, _ = conv
        collector = trace_collector.start(
            conversation_id=1, message_id=1, user_id=1, agent_code=None, model_id="m"
        )
        try:
            await converter.record_plan({"tasks": [{"id": "1", "description": "d"}]}, phase="plan")
            assert collector.context_events[-1]["event"] == "plan"
            assert collector.context_events[-1]["phase"] == "plan"
        finally:
            trace_collector._current_collector.set(None)
