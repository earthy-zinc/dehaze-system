from types import SimpleNamespace

import pytest

from app.service import ai_message_service as m


def _conv(agent_code="default", model_config=None):
    return SimpleNamespace(agent_code=agent_code, model_config=model_config or {})


def _agent(reasoning_mode="auto"):
    return SimpleNamespace(reasoning_mode=reasoning_mode)


def _patch_agent(monkeypatch, reasoning_mode):
    async def get_by_code(db, code):
        return None if reasoning_mode is None else _agent(reasoning_mode)

    monkeypatch.setattr(m.ai_agent_repository, "get_by_code", get_by_code)


class TestNeedsToolCall:

    async def test_defaults_true_when_no_config(self, monkeypatch):
        _patch_agent(monkeypatch, None)
        assert await m._needs_tool_call(object(), _conv()) is True

    async def test_explicit_need_tools_false_wins(self, monkeypatch):
        _patch_agent(monkeypatch, "react")
        conv = _conv(model_config={"needTools": False})
        assert await m._needs_tool_call(object(), conv) is False

    async def test_direct_agent_skips_validation(self, monkeypatch):
        _patch_agent(monkeypatch, "direct")
        assert await m._needs_tool_call(object(), _conv()) is False

    async def test_non_direct_agent_defaults_true(self, monkeypatch):
        _patch_agent(monkeypatch, "auto")
        assert await m._needs_tool_call(object(), _conv()) is True
