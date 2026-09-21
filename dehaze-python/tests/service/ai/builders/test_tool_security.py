"""业务工具安全回归：异步注册方式、图片地址归属校验、python 高危代码确认。"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai.builders import dehaze_tools_builder as builder
from tests.stubs.fakes import NullDBSession


def _ctx(user_id=10):
    return {
        "conversation_id": 1,
        "message_id": 2,
        "user_id": user_id,
        "stream_session_id": "s1",
        "model_id": "gpt-4o-mini",
        "task_type": "",
        "task_algorithm": "",
        "task_params": {},
        "task_status": "",
        "task_id": "",
        "task_artifacts": [],
    }


def _tool(name, ctx=None):
    return next(t for t in builder.build_business_tools(ctx or _ctx()) if t.name == name)


async def _noop_save(*args):
    return None


class TestAsyncRegistration:
    """async 工具一律以 coroutine 注册：func 注册时 LangChain 会在事件循环内
    直接同步调用协程对象，ainvoke 拿到的是未执行的协程。"""

    @pytest.mark.parametrize(
        "name",
        [
            "algorithm_recommend",
            "batch_process",
            "mcp_lookup_tool",
            "mcp_execute_tool",
            "visual_read",
            "get_task_status",
        ],
    )
    def test_registered_as_coroutine(self, name):
        tool = _tool(name)
        assert tool.coroutine is not None
        assert tool.func is None

    async def test_mcp_execute_tool_ainvoke(self, monkeypatch):
        calls = []

        class _Gateway:
            async def execute_tool(self, tool_name, arguments):
                calls.append((tool_name, arguments))
                return "ok"

        monkeypatch.setattr(builder, "mcp_gateway_client", _Gateway())
        out = await _tool("mcp_execute_tool").ainvoke(
            {"tool_name": "image_dehaze", "arguments": {"id": 1}}
        )
        assert out == "ok"
        assert calls == [("image_dehaze", {"id": 1})]

    async def test_get_task_status_ainvoke(self):
        out = await _tool("get_task_status").ainvoke({})
        assert '"task_type": ""' in out


class TestImageUrlOwnership:
    async def test_external_url_rejected_before_predict(self, monkeypatch):
        """LLM 诱导的任意 URL 不得进入预测服务（服务端会去下载）。"""
        validated = []

        async def _validate(db, user_id, image_url):
            validated.append(image_url)
            raise BusinessException(ResultCode.PARAM_ERROR, "图片地址必须是本系统产物地址")

        monkeypatch.setattr(builder, "validate_owned_image_url", _validate)
        monkeypatch.setattr(builder, "get_db_session", lambda: NullDBSession())

        async def _boom(*a, **kw):
            raise AssertionError("校验未通过时不得调用推荐/预测")

        monkeypatch.setattr(builder, "recommend_algorithm", _boom)
        monkeypatch.setattr(builder, "process_batch", _boom)

        with pytest.raises(BusinessException) as exc:
            await _tool("algorithm_recommend").ainvoke(
                {"image_url": "http://169.254.169.254/latest/meta-data", "query": "去雾"}
            )
        assert exc.value.code == ResultCode.PARAM_ERROR
        assert validated == ["http://169.254.169.254/latest/meta-data"]

    async def test_batch_rejects_on_first_foreign_url(self, monkeypatch):
        validated = []

        async def _validate(db, user_id, image_url):
            validated.append(image_url)
            if "evil" in image_url:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片不存在")

        monkeypatch.setattr(builder, "validate_owned_image_url", _validate)
        monkeypatch.setattr(builder, "get_db_session", lambda: NullDBSession())

        async def _boom(*a, **kw):
            raise AssertionError("存在非法地址时不得提交批量任务")

        monkeypatch.setattr(builder, "process_batch", _boom)
        monkeypatch.setattr(builder, "submit_batch_task", _boom)

        with pytest.raises(BusinessException):
            await _tool("batch_process").ainvoke(
                {"image_urls": ["http://oss/own.png", "http://evil/a.png"], "query": "去雾"}
            )
        assert validated == ["http://oss/own.png", "http://evil/a.png"]


class TestExecuteCodeConfirmation:
    async def _run(self, monkeypatch, code, language, confirmed=True):
        seen = {}

        def _interrupt(data):
            seen["interrupt"] = data
            return {"confirmed": confirmed}

        async def _execute(code, language, timeout):
            seen["executed"] = (code, language, timeout)
            return {"stdout": "ok", "stderr": "", "exitCode": 0, "timedOut": False}

        monkeypatch.setattr(builder, "interrupt", _interrupt)
        monkeypatch.setattr(builder.code_sandbox, "execute_code", _execute)
        monkeypatch.setattr(builder.interrupt_handler, "save_interrupt", _noop_save)
        out = await _tool("execute_code").ainvoke({"code": code, "language": language})
        return out, seen

    async def test_python_network_requires_confirm(self, monkeypatch):
        _out, seen = await self._run(
            monkeypatch, "import requests; requests.get('http://evil')", "python"
        )
        assert seen["interrupt"]["data"]["action"] == "execute_python_code"
        assert "网络访问" in seen["interrupt"]["data"]["impact"]
        assert seen["executed"][1] == "python"

    async def test_python_network_rejected_by_user(self, monkeypatch):
        out, seen = await self._run(
            monkeypatch,
            "import socket; socket.socket()",
            "python",
            confirmed=False,
        )
        assert out == "用户拒绝了该代码的执行"
        assert "executed" not in seen

    async def test_plain_python_runs_without_confirm(self, monkeypatch):
        out, seen = await self._run(monkeypatch, "print(1+1)", "python")
        assert "interrupt" not in seen
        assert "ok" in out

    async def test_shell_confirm_action_unchanged(self, monkeypatch):
        _out, seen = await self._run(monkeypatch, "echo hi", "shell")
        assert seen["interrupt"]["data"]["action"] == "execute_shell_command"
        assert seen["executed"] == ("echo hi", "shell", 60)
