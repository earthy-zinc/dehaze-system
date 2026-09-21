import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.infrastructure.sse.sse_event_converter import SseEventConverter
from app.service.ai.service import trace_collector
from tests.stubs.fakes import RecorderEmitter


@pytest.fixture
def conv(monkeypatch):
    ctx = {
        "stream_session_id": "s1",
        "message_id": 1,
        "conversation_id": 100,
    }
    emitter = RecorderEmitter()
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter)

    async def _noop_create(db, **kw):
        return None

    monkeypatch.setattr(
        "app.infrastructure.sse.sse_event_converter.ai_agent_thought_repository",
        type("R", (), {"create_thought": staticmethod(_noop_create)})(),
    )

    return SseEventConverter(ctx), emitter


class TestEventMapping:
    async def test_text_delta(self, conv):
        converter, emitter = conv
        await converter.handle(
            {
                "type": "messages",
                "ns": [],
                "data": [AIMessageChunk(content="你好")],
            }
        )
        delta = next(e for e in emitter.events if e[0] == "content_block.delta")
        assert delta[1]["delta"] == {"type": "text_delta", "text": "你好"}

    async def test_content_block_start_before_first_text_delta(self, conv):
        converter, emitter = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="第一段")]}
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="第二段")]}
        )
        assert emitter.events[0][0] == "content_block.start"
        assert emitter.events[0][1] == {"index": 0, "type": "text"}
        starts = [e for e in emitter.events if e[0] == "content_block.start"]
        assert len(starts) == 1
        assert emitter.events[1][0] == "content_block.delta"

    async def test_content_block_stop_on_finish(self, conv):
        converter, emitter = conv
        await converter.finish()
        assert not [e for e in emitter.events if e[0] == "content_block.stop"]
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="内容")]}
        )
        await converter.finish()
        assert emitter.events[-1] == ("content_block.stop", {"index": 0})

    async def test_content_block_start_not_for_tool_delta(self, conv):
        converter, emitter = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {"name": "search", "args": '{"query":', "id": "c1", "index": 0},
        ]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        assert not [e for e in emitter.events if e[0] == "content_block.start"]

    async def test_tool_call_delta(self, conv):
        converter, emitter = conv
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {"name": "search", "args": '{"query":', "id": "c1", "index": 0},
        ]
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        assert emitter.events[0][0] == "content_block.delta"
        assert emitter.events[0][1]["delta"]["type"] == "input_json_delta"

    async def test_thought_on_tool_message(self, conv):
        converter, emitter = conv
        chunk1 = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "search", "args": '{"query":', "id": "c1", "index": None}],
        )
        chunk2 = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": '"去雾"}', "id": "c1", "index": None}],
        )
        await converter.handle({"type": "messages", "ns": [], "data": [chunk1]})
        await converter.handle({"type": "messages", "ns": [], "data": [chunk2]})
        await converter.handle(
            {
                "type": "updates",
                "ns": ["subagent"],
                "data": {
                    "tool_node": {
                        "messages": [
                            ToolMessage(content="ok", tool_call_id="c1", name="search"),
                        ]
                    }
                },
            }
        )
        thought = [e for e in emitter.events if e[0] == "thought"]
        assert thought, "应推送 thought"
        assert thought[0][1]["tool"] == "search"
        assert thought[0][1]["toolInput"] == {"query": "去雾"}
        assert thought[0][1]["latencyMs"] >= 0
        assert thought[0][1]["position"] == 1

    async def test_custom_event(self, conv):
        converter, emitter = conv
        await converter.handle({"type": "custom", "ns": [], "data": {"k": 1}})
        assert emitter.events[0][0] == "custom"
        assert emitter.events[0][1]["data"] == {"k": 1}

    async def test_unknown_type_ignored(self, conv):
        converter, emitter = conv
        await converter.handle({"type": "values", "ns": [], "data": {}})
        assert emitter.events == []

    async def test_no_stream_session_skips_emit(self, monkeypatch):
        converter = SseEventConverter({"stream_session_id": None})
        emitter = RecorderEmitter()
        monkeypatch.setattr(
            "app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter
        )
        await converter.handle({"type": "custom", "ns": [], "data": {"x": 1}})
        assert emitter.events == []


class TestInterrupt:
    async def test_interrupt_mapping_contract(self, conv):
        from langgraph.types import Interrupt

        converter, emitter = conv
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "type": "confirm",
                                "data": {"artifactId": 1, "recommendation": {"algorithmName": "A"}},
                            }
                        )
                    ]
                },
            }
        )
        assert emitter.events[0] == (
            "interrupt",
            {
                "type": "confirm",
                "data": {"artifactId": 1, "recommendation": {"algorithmName": "A"}},
            },
        )

    async def test_interrupt_defaults_type_when_absent(self, conv):
        from langgraph.types import Interrupt

        converter, emitter = conv
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {"__interrupt__": [Interrupt(value={"foo": "bar"})]},
            }
        )
        assert emitter.events[0] == ("interrupt", {"type": "confirm", "data": {"foo": "bar"}})

    async def test_plan_approve_payload_normalized(self, conv):
        """plan_approve 的 plan 在中断点内是内部 snake_case，仅在 SSE 出口归一，
        且不带 phase（phase 只属于 plan 事件）。"""
        from langgraph.types import Interrupt

        converter, emitter = conv
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "type": "plan_approve",
                                "stream_session_id": "s1",
                                "data": {
                                    "plan": {
                                        "tasks": [
                                            {
                                                "id": "A",
                                                "description": "分析",
                                                "depends_on": ["B"],
                                                "tool_hint": "analyze",
                                                "paradigm": "react",
                                                "status": "pending",
                                                "result": None,
                                            }
                                        ],
                                        "status": "pending",
                                        "revisions": [
                                            {
                                                "revisionNo": 1,
                                                "reason": "B",
                                                "changedTaskIds": ["B2"],
                                            }
                                        ],
                                    }
                                },
                            }
                        )
                    ]
                },
            }
        )
        assert emitter.events[0] == (
            "interrupt",
            {
                "type": "plan_approve",
                "data": {
                    "plan": {
                        "tasks": [
                            {
                                "id": "A",
                                "description": "分析",
                                "dependsOn": ["B"],
                                "status": "pending",
                                "paradigm": "react",
                                "toolHint": "analyze",
                            }
                        ],
                        "status": "pending",
                        "revisions": [{"revisionNo": 1, "reason": "B", "changedTaskIds": ["B2"]}],
                    }
                },
            },
        )


class TestPiiMaskingAtStreamExit:
    """PII 脱敏必须在 SSE 推送前完成：事后脱敏等于已经把敏感信息推给了客户端。"""

    @staticmethod
    def _texts(emitter):
        return "".join(
            e[1]["delta"]["text"]
            for e in emitter.events
            if e[0] == "content_block.delta" and e[1]["delta"].get("type") == "text_delta"
        )

    async def test_phone_split_across_chunks_masked(self, conv):
        converter, emitter = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="联系138")]}
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="12345678，谢谢")]}
        )
        await converter.finish()
        text = self._texts(emitter)
        assert "13812345678" not in text
        assert text == "联系***，谢谢"

    async def test_secret_key_masked(self, conv):
        converter, emitter = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="key: sk-abcdef123456")]}
        )
        await converter.finish()
        assert self._texts(emitter) == "key: ***"

    async def test_thinking_channel_masked(self, conv):
        converter, emitter = conv
        chunk = AIMessageChunk(content="", additional_kwargs={"thinking": "手机号13812345678"})
        await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
        await converter.finish()
        thinking = "".join(
            e[1]["delta"]["thinking"]
            for e in emitter.events
            if e[0] == "content_block.delta" and e[1]["delta"].get("type") == "thinking_delta"
        )
        assert "13812345678" not in thinking
        assert thinking == "手机号***"

    async def test_plain_text_not_delayed(self, conv):
        """无敏感串前缀时不得压住不推送（尾部保留窗口只用于跨 chunk 的敏感串）。"""
        converter, emitter = conv
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="你好")]}
        )
        assert self._texts(emitter) == "你好"


class TestSubagentAttribution:
    """子 Agent/Team 推理步骤归属：ns 非空即子图事件，agent_code 取末段节点名"""

    async def _capture_create(self, monkeypatch):
        captured = []

        async def _capture(db, **kw):
            captured.append(kw)
            return

        monkeypatch.setattr(
            "app.infrastructure.sse.sse_event_converter.ai_agent_thought_repository",
            type("R", (), {"create_thought": staticmethod(_capture)})(),
        )
        return captured

    async def test_subagent_thought_attribution(self, conv, monkeypatch):
        converter, _emitter = conv
        captured = await self._capture_create(monkeypatch)
        # 子图事件（ns 非空，段格式"节点名:task_id"）：归属子 Agent
        await converter.handle(
            {
                "type": "updates",
                "ns": ["task:abc123"],
                "data": {
                    "tool_node": {
                        "messages": [ToolMessage(content="ok", tool_call_id="c9", name="search")]
                    }
                },
            }
        )
        assert captured[-1]["agent_code"] == "task"
        assert captured[-1]["is_subagent"] == 1
        # 主图事件（ns 空）：归属主 Agent
        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "tool_node": {
                        "messages": [ToolMessage(content="ok", tool_call_id="c10", name="search")]
                    }
                },
            }
        )
        assert captured[-1]["agent_code"] is None
        assert captured[-1]["is_subagent"] == 0

    async def test_nested_subgraph_takes_last_segment(self, conv, monkeypatch):
        """嵌套子图 ns 多段，取最后一段去 task_id 后缀"""
        converter, _emitter = conv
        captured = await self._capture_create(monkeypatch)
        await converter.handle(
            {
                "type": "updates",
                "ns": ["task:a", "inner:b"],
                "data": {
                    "tool_node": {
                        "messages": [ToolMessage(content="ok", tool_call_id="c11", name="search")]
                    }
                },
            }
        )
        assert captured[-1]["agent_code"] == "inner"
        assert captured[-1]["is_subagent"] == 1

    async def test_plan_recorded_to_collector(self, conv):
        converter, _emitter = conv
        collector = trace_collector.start(
            conversation_id=1, message_id=1, user_id=1, agent_code=None, model_id="m"
        )
        try:
            await converter.record_plan({"tasks": [{"id": "1", "description": "d"}]}, phase="plan")
            assert collector.context_events[-1]["event"] == "plan"
            assert collector.context_events[-1]["phase"] == "plan"
        finally:
            trace_collector._current_collector.set(None)


class TestRecordPlanTaskFields:
    """plan 事件任务项：既有字段保口径，paradigm/toolHint/result 补齐且缺失不下发。"""

    async def _plan_task(self, conv, task):
        converter, emitter = conv
        await converter.record_plan({"tasks": [task]}, phase="plan")
        event = next(e for e in emitter.events if e[0] == "plan")[1]
        return event["tasks"][0]

    async def test_optional_fields_present_when_provided(self, conv):
        task = await self._plan_task(
            conv,
            {
                "id": "A",
                "description": "分析",
                "depends_on": [],
                "paradigm": "reflexion",
                "tool_hint": "analyze",
                "result": "结论",
                "status": "done",
            },
        )
        assert task["id"] == "A"
        assert task["dependsOn"] == []
        assert task["status"] == "done"
        assert task["paradigm"] == "reflexion"
        assert task["toolHint"] == "analyze"
        assert task["result"] == "结论"

    async def test_missing_optional_fields_omitted(self, conv):
        """plan task 缺 paradigm/tool_hint/result：整个键不下发（不塞 None）。"""
        task = await self._plan_task(conv, {"id": "A", "description": "分析"})
        assert set(task.keys()) == {"id", "description", "dependsOn", "status"}
        assert task["dependsOn"] == []
        assert task["status"] == "pending"

    async def test_none_optional_fields_omitted(self, conv):
        """可选字段显式为 None 时同样不下发，既有 dependsOn 不受影响。"""
        task = await self._plan_task(
            conv,
            {
                "id": "A",
                "description": "分析",
                "depends_on": ["X"],
                "paradigm": None,
                "tool_hint": None,
                "result": None,
            },
        )
        assert "paradigm" not in task
        assert "toolHint" not in task
        assert "result" not in task
        assert task["dependsOn"] == ["X"]

    async def test_empty_string_result_emitted(self, conv):
        """空串 result 非 None：按有值下发（'null 不下发'仅针对 None）。"""
        task = await self._plan_task(conv, {"id": "A", "description": "分析", "result": ""})
        assert task["result"] == ""

    async def test_no_tasks_emits_empty_list(self, conv):
        converter, emitter = conv
        await converter.record_plan({}, phase="plan")
        event = next(e for e in emitter.events if e[0] == "plan")[1]
        assert event["tasks"] == []
        assert event["status"] == "pending"
