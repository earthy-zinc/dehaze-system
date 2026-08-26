from unittest.mock import AsyncMock

from app.service.ai.builders import dehaze_tools_builder as builder
from app.infrastructure.sandbox.code_sandbox import CodeSandbox
from app.service.ai.builders.knowledge_base_tool import KnowledgeBaseClient


class _FakeWeb:
    def __init__(self, results=None):
        self._results = results

    async def search(self, query, max_results=8):
        return self._results


class _FakeKB:
    def __init__(self, results=None):
        self._results = results if results is not None else []

    async def retrieve(self, query, top_k=5, user_id=None):
        return self._results

    format_results = KnowledgeBaseClient.format_results


_KB_RESULTS = [
    {"title": "去雾算法综述", "snippet": "综述了主流去雾算法", "source": "KB-1"},
    {"title": "RIDCP 论文", "snippet": "基于扩散的去雾", "source": "KB-2"},
]

_WEB_RESULTS = [
    {"title": f"结果{i}", "url": f"https://e.com/{i}", "snippet": f"摘要{i}"} for i in range(8)
]


def _patch_web_search(monkeypatch, *, quota=True, client=None):
    monkeypatch.setattr(builder, "get_redis_client", AsyncMock(return_value=object()))
    monkeypatch.setattr(builder, "check_search_quota", AsyncMock(return_value=quota))
    monkeypatch.setattr(
        builder, "web_search_client", client if client is not None else _FakeWeb(_WEB_RESULTS)
    )


def _patch_kb_client(monkeypatch, results=_KB_RESULTS):
    monkeypatch.setattr(builder, "knowledge_base_client", _FakeKB(results))


def _make_ctx():
    return {
        "conversation_id": 1,
        "message_id": 2,
        "user_id": 10,
        "stream_session_id": "s1",
        "model_id": "gpt-4o-mini",
        "task_type": "",
        "task_algorithm": "",
        "task_params": {},
        "task_status": "",
        "task_id": "",
        "task_artifacts": [],
    }


def _tools(ctx):
    return builder.build_business_tools(ctx)


def _get_tool(ctx, name):
    return next(t for t in _tools(ctx) if t.name == name)


def _make_interrupt(confirmed=True, recorder=None):
    def _fake(payload):
        if recorder is not None:
            recorder["interrupt"] = payload
        return {"confirmed": confirmed}

    return _fake


async def test_web_search_returns_limited_and_formatted(monkeypatch):
    _patch_web_search(monkeypatch)
    ctx = _make_ctx()
    tool = _get_tool(ctx, "web_search")
    out = await tool.ainvoke({"query": "q", "max_results": 10})
    assert out.count("来源: https://e.com/") == 8
    assert "1. 结果0" in out


async def test_web_search_clamps_max_results(monkeypatch):
    seen = {}

    class _Fake:
        async def search(self, query, max_results=8):
            seen["mr"] = max_results
            return [{"title": "x", "url": "u", "snippet": "s"}] * max_results

    _patch_web_search(monkeypatch, client=_Fake())
    ctx = _make_ctx()
    tool = _get_tool(ctx, "web_search")
    await tool.ainvoke({"query": "q", "max_results": 100})
    assert seen["mr"] == 10
    await tool.ainvoke({"query": "q", "max_results": 2})
    assert seen["mr"] == 5


async def test_web_search_quota_exhausted_degrades(monkeypatch):
    _patch_web_search(monkeypatch, quota=False)
    _patch_kb_client(monkeypatch)
    ctx = _make_ctx()
    out = await _get_tool(ctx, "web_search").ainvoke({"query": "q"})
    assert out.startswith("搜索配额已用尽，可尝试知识库检索。已降级为知识库检索：")
    assert "去雾算法综述" in out


async def test_web_search_unavailable_degrades_to_kb(monkeypatch):
    _patch_web_search(monkeypatch, client=_FakeWeb(None))
    _patch_kb_client(monkeypatch)
    ctx = _make_ctx()
    out = await _get_tool(ctx, "web_search").ainvoke({"query": "q"})
    assert out.startswith("网络搜索不可用，已降级为知识库检索：")
    assert "去雾算法综述" in out


async def test_web_search_degrades_to_empty_kb(monkeypatch):
    _patch_web_search(monkeypatch, client=_FakeWeb(None))
    _patch_kb_client(monkeypatch, results=[])
    ctx = _make_ctx()
    out = await _get_tool(ctx, "web_search").ainvoke({"query": "q"})
    assert out.endswith("知识库暂无可检索内容")


async def test_knowledge_base_empty(monkeypatch):
    _patch_kb_client(monkeypatch, results=[])
    ctx = _make_ctx()
    out = await _get_tool(ctx, "knowledge_base_search").ainvoke({"query": "q", "top_k": 5})
    assert out == "知识库暂无可检索内容"


async def test_knowledge_base_returns_with_source(monkeypatch):
    _patch_kb_client(monkeypatch)
    ctx = _make_ctx()
    out = await _get_tool(ctx, "knowledge_base_search").ainvoke({"query": "q", "top_k": 2})
    assert "去雾算法综述" in out
    assert "来源: KB-1" in out


async def test_sandbox_normal_python():
    sb = CodeSandbox()
    result = await sb.execute_code("print(42)", "python", timeout=10)
    assert result["exitCode"] == 0
    assert "42" in result["stdout"]


async def test_sandbox_timeout_terminates():
    sb = CodeSandbox()
    result = await sb.execute_code("import time; time.sleep(5)", "python", timeout=1)
    assert result["timedOut"] is True
    assert "执行超时(1s)已终止" in result["stderr"]


async def test_sandbox_blacklist_rejects():
    sb = CodeSandbox()
    result = await sb.execute_code("rm -rf /tmp/x", "shell", timeout=10)
    assert result["exitCode"] == 1
    assert "已拒绝执行" in result["stderr"]
    assert sb.check_blacklist("rm -rf /tmp/x") is not None
    assert sb.check_blacklist("ls -la") is None


async def test_sandbox_unsupported_language():
    sb = CodeSandbox()
    result = await sb.execute_code("x=1", "java", timeout=10)
    assert "不支持的语言: java" in result["stderr"]
    assert result["exitCode"] == 1


async def test_sandbox_output_path_desensitized():
    sb = CodeSandbox()
    result = await sb.execute_code(
        "import os,sys; print(os.getcwd()); sys.stderr.write(os.getcwd())", "python", timeout=10
    )
    assert "/tmp" not in result["stdout"]
    assert "/workspace" in result["stdout"]
    assert "/tmp" not in result["stderr"]
    assert "/workspace" in result["stderr"]


async def test_sandbox_output_truncated():
    sb = CodeSandbox(output_limit=100)
    result = await sb.execute_code("print('x' * 5000)", "python", timeout=10)
    assert result["truncated"]["stdout"] is True
    assert "输出已截断，共" in result["stdout"]
    assert len(result["stdout"]) < 200


async def test_execute_code_shell_interrupt_confirm(monkeypatch):
    called = {"interrupt": None, "executed": None}

    async def _fake_execute(code, language, timeout):
        called["executed"] = (code, language, timeout)
        return {"stdout": "ok", "stderr": "", "exitCode": 0, "timedOut": False}

    monkeypatch.setattr(builder, "interrupt", _make_interrupt(recorder=called))
    monkeypatch.setattr(builder.code_sandbox, "execute_code", _fake_execute)
    ctx = _make_ctx()
    out = await _get_tool(ctx, "execute_code").ainvoke(
        {"code": "echo hi", "language": "shell", "timeout": 30}
    )
    assert called["interrupt"]["type"] == "confirm"
    assert called["interrupt"]["data"]["command"] == "echo hi"
    assert called["executed"] == ("echo hi", "shell", 30)
    assert "ok" in out


async def test_execute_code_shell_user_rejects(monkeypatch):
    executed = {"hit": False}

    async def _fake_execute(*a, **k):
        executed["hit"] = True

    monkeypatch.setattr(builder, "interrupt", _make_interrupt(confirmed=False))
    monkeypatch.setattr(builder.code_sandbox, "execute_code", _fake_execute)
    ctx = _make_ctx()
    out = await _get_tool(ctx, "execute_code").ainvoke(
        {"code": "echo hi", "language": "shell", "timeout": 30}
    )
    assert "用户拒绝了该 Shell 命令的执行" in out
    assert executed["hit"] is False


async def test_execute_code_shell_blacklist_no_interrupt(monkeypatch):
    called = {"interrupt": False}
    monkeypatch.setattr(builder, "interrupt", _make_interrupt(recorder=called))
    ctx = _make_ctx()
    out = await _get_tool(ctx, "execute_code").ainvoke(
        {"code": "shutdown -h now", "language": "shell", "timeout": 30}
    )
    assert "已拒绝执行" in out
    assert called["interrupt"] is False


async def test_three_tools_ainvoke_smoke(monkeypatch):
    _patch_web_search(monkeypatch)
    _patch_kb_client(monkeypatch)
    monkeypatch.setattr(builder, "interrupt", _make_interrupt())
    ctx = _make_ctx()
    tools = {t.name: t for t in _tools(ctx)}

    ws = await tools["web_search"].ainvoke({"query": "最新去雾算法"})
    assert "来源: https://e.com/" in ws

    kb = await tools["knowledge_base_search"].ainvoke({"query": "去雾", "top_k": 2})
    assert "去雾算法综述" in kb

    ec = await tools["execute_code"].ainvoke({"code": "print(1+1)", "language": "python"})
    assert "2" in ec
