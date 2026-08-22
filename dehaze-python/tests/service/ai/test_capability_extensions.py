from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.service.ai.capability_constraints import CapabilityConstraintsMiddleware
from app.service.ai.mcp_namespace_prefilter import (
    McpNamespacePrefilter,
    McpNamespacePrefilterMiddleware,
)
from app.service.ai.tool_failure_guard import ToolFailureGuardMiddleware


class _FakeGateway:
    def __init__(self, tools=None, unavailable=False):
        self.tools = tools or []
        self.unavailable = unavailable
        self.list_calls = 0

    async def list_tools(self):
        self.list_calls += 1
        return [] if self.unavailable else self.tools

    async def lookup_tool_param_schema(self, tool_name):
        return f"{tool_name} __FULL_SCHEMA__"


def _image_tools():
    return [
        {
            "name": "image_processing_dehaze",
            "description": "图像去雾处理",
            "input_schema": {"properties": {"image": {}, "strength": {}}},
        },
        {
            "name": "evaluation_psnr",
            "description": "PSNR 指标评估",
            "input_schema": {"properties": {"image": {}}},
        },
    ]


class _ToolRequest:
    def __init__(self, name, args=None, state=None, tool_call_id="call_1"):
        self.tool_call = {"name": name, "args": args or {}, "id": tool_call_id}
        self.state = state


async def _fail(req):
    raise RuntimeError("x")


async def _ok(req):
    return ToolMessage(content="ok", tool_call_id="call_1")


class _State:
    def __init__(self, files=None):
        self.files = files or {}


async def _run_tool(mw, name, args, state):
    called = []

    async def handler(r):
        called.append(1)
        return ToolMessage(content="ok", tool_call_id="call_1")

    msg = await mw.awrap_tool_call(_ToolRequest(name, args, state), handler)
    return msg, called


class TestNamespaceMatching:
    def test_chinese_hit(self):
        p = McpNamespacePrefilter()
        assert "image_processing" in p.match_namespaces("帮这张图去雾")
        assert "evaluation" in p.match_namespaces("评估一下 PSNR 指标")

    def test_english_hit(self):
        p = McpNamespacePrefilter()
        assert "image_processing" in p.match_namespaces("defog this image")

    def test_no_match_returns_empty(self):
        p = McpNamespacePrefilter()
        assert p.match_namespaces("今天天气怎么样") == []

    def test_matches_at_most_two(self):
        p = McpNamespacePrefilter()
        matched = p.match_namespaces("去雾并评估 PSNR，再跑批量处理")
        assert len(matched) <= 2
        assert matched[0] == "image_processing"

    async def test_agent_namespace_filter(self):
        p = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        block = await p.build_tools_block("去雾并评估", agent_namespaces=["evaluation"])
        assert "evaluation_psnr" in block
        assert "image_processing_dehaze" not in block

    async def test_agent_namespace_no_match_falls_to_fuzzy(self):
        p = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        block = await p.build_tools_block("去雾", agent_namespaces=["dataset"])
        assert "mcp_lookup_tool" in block


class TestNamespaceCache:
    async def test_cache_hit(self):
        g = _FakeGateway(_image_tools())
        p = McpNamespacePrefilter(g)
        await p.list_namespaces()
        await p.list_namespaces()
        assert g.list_calls == 1

    async def test_cache_expiry(self):
        g = _FakeGateway(_image_tools())
        p = McpNamespacePrefilter(g)
        await p.list_namespaces()
        p._cache["fetched_at"] -= settings.AI_MCP_NAMESPACE_CACHE_TTL + 1
        await p.list_namespaces()
        assert g.list_calls == 2

    async def test_gateway_unavailable_degradation(self):
        p = McpNamespacePrefilter(_FakeGateway(unavailable=True))
        summaries = await p.list_namespaces()
        assert summaries == {}

    async def test_namespace_summary_structure(self):
        p = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        summaries = await p.list_namespaces()
        assert summaries["image_processing"]["tool_count"] == 1
        assert summaries["evaluation"]["tool_count"] == 1
        assert summaries["image_processing"]["name"] == "image_processing"


class TestBuildToolsBlock:
    async def test_hit_expand_injects_full_definitions(self):
        p = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        block = await p.build_tools_block("帮这张图去雾")
        assert "image_processing_dehaze" in block
        assert "图像去雾处理" in block
        assert "参数定义：" in block
        assert "image_processing_dehaze __FULL_SCHEMA__" in block

    async def test_no_match_injects_fuzzy_fallback(self):
        p = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        block = await p.build_tools_block("今天天气怎么样")
        assert "mcp_lookup_tool" in block
        assert "命名空间摘要" in block

    async def test_gateway_down_injects_guidance(self):
        p = McpNamespacePrefilter(_FakeGateway(unavailable=True))
        block = await p.build_tools_block("帮这张图去雾")
        assert "mcp_lookup_tool" in block


class TestPrefilterMiddleware:
    async def test_awakens_and_injects_to_system_message(self):
        state = {"messages": [HumanMessage(content="帮这张图去雾")]}
        prefilter = McpNamespacePrefilter(_FakeGateway(_image_tools()))
        mw = McpNamespacePrefilterMiddleware(agent_namespaces=None, prefilter=prefilter)
        request = ModelRequest(
            model=None,
            messages=state["messages"],
            system_message=SystemMessage(content="基础提示"),
            tools=[],
            state=state,
            runtime=None,
        )
        seen = {}

        async def handler(req):
            seen["sys"] = req.system_message.content
            return "resp"

        result = await mw.awrap_model_call(request, handler)
        assert result == "resp"
        assert "基础提示" in seen["sys"]
        assert "image_processing_dehaze" in seen["sys"]


class TestToolFailureGuard:
    async def test_three_consecutive_failures_disables_tool(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)
        last = None
        for _ in range(3):
            last = await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        assert "algo_recommend" in mw._disabled[0]
        assert "临时禁用" in last.content

    async def test_tool_blocked_after_disabled(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)
        for _ in range(3):
            await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        called = []

        async def blocked(req):
            called.append(1)
            return ToolMessage(content="ok", tool_call_id="call_1")

        msg = await mw.awrap_tool_call(_ToolRequest("algo_recommend"), blocked)
        assert called == []
        assert "临时禁用" in msg.content

    async def test_success_resets_counter(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)
        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _ok)
        assert mw._fails[0]["algo_recommend"] == 0
        assert "algo_recommend" not in mw._disabled.get(0, set())

    async def test_new_run_resets_counter(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)
        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), _fail)
        await mw.abefore_agent({"messages": []}, object())
        assert mw._fails == {}
        assert mw._disabled == {}

    async def test_error_status_toolmessage_counts_as_failure(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)

        async def err(req):
            return ToolMessage(content="工具参数有误", tool_call_id="call_1", status="error")

        await mw.awrap_tool_call(_ToolRequest("algo_recommend"), err)
        assert mw._fails[0]["algo_recommend"] == 1

    async def test_concurrent_conversations_isolated(self):
        mw = ToolFailureGuardMiddleware(fail_limit=3)
        for _ in range(3):
            await mw.awrap_tool_call(_ToolRequest("algo_recommend", state={"conversation_id": 1}), _fail)
        assert "algo_recommend" in mw._disabled[1]

        msg = await mw.awrap_tool_call(_ToolRequest("algo_recommend", state={"conversation_id": 2}), _ok)
        assert msg.content == "ok"

        await mw.abefore_agent({"conversation_id": 2, "messages": []}, object())
        assert "algo_recommend" in mw._disabled[1]
        assert 2 not in mw._fails


class TestWriteFileCapacity:
    async def test_over_capacity_blocked(self, monkeypatch):
        monkeypatch.setattr(settings, "AI_VFS_MAX_BYTES", 4096)
        mw = CapabilityConstraintsMiddleware()
        existing = {"a.txt": {"content": "x" * 4095}}
        msg, called = await _run_tool(
            mw,
            "write_file",
            {"file_path": "/ws/b.txt", "content": "y" * 100},
            _State(existing),
        )
        assert called == []
        assert "容量超限" in msg.content
        assert msg.status == "error"

    async def test_within_capacity_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "AI_VFS_MAX_BYTES", 4096)
        mw = CapabilityConstraintsMiddleware()
        existing = {"a.txt": {"content": "去雾缓存"}}
        msg, called = await _run_tool(
            mw,
            "write_file",
            {"file_path": "/ws/b.txt", "content": "world"},
            _State(existing),
        )
        assert called == [1]
        assert msg.content == "ok"

    async def test_overwrite_existing_file_does_not_double_count(self, monkeypatch):
        monkeypatch.setattr(settings, "AI_VFS_MAX_BYTES", 4096)
        mw = CapabilityConstraintsMiddleware()
        existing = {"a.txt": {"content": "x" * 4000}}
        msg, called = await _run_tool(
            mw,
            "write_file",
            {"file_path": "/ws/a.txt", "content": "y" * 10},
            _State(existing),
        )
        assert called == [1]

    async def test_other_tools_pass_through(self):
        mw = CapabilityConstraintsMiddleware()
        msg, called = await _run_tool(mw, "ls", {"path": "/ws"}, _State({}))
        assert called == [1]


class TestWriteTodos:
    async def test_over_32_items_blocked(self):
        mw = CapabilityConstraintsMiddleware()
        todos = [{"content": f"第{i}项任务", "status": "pending"} for i in range(33)]
        msg, called = await _run_tool(mw, "write_todos", {"todos": todos}, _State({}))
        assert called == []
        assert "32" in msg.content
        assert msg.status == "error"

    async def test_within_32_items_passes(self):
        mw = CapabilityConstraintsMiddleware()
        todos = [{"content": f"第{i}项任务", "status": "pending"} for i in range(10)]
        msg, called = await _run_tool(mw, "write_todos", {"todos": todos}, _State({}))
        assert called == [1]

    async def test_overlong_item_warns_but_passes(self):
        mw = CapabilityConstraintsMiddleware()
        long_desc = "这是一个非常长的任务描述" * 20
        todos = [{"content": long_desc, "status": "pending"}]
        msg, called = await _run_tool(mw, "write_todos", {"todos": todos}, _State({}))
        assert called == [1]
        assert msg.content == "ok"
