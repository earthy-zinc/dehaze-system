"""子 Agent 生命周期钩子分流：记忆提取与会话标题更新仅归属主图 run

口径（AI对话 多步推理 / 智能体管理 §子 Agent）：
- 子图 run 的 after_agent 不做记忆提取（子图消息源于任务描述，
  提取入长期记忆会污染用户画像）；
- 子图 run 的 after_agent 不更新主会话标题（标题口径归属主会话首条对话）；
- 主图 run 行为不变。
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.service.ai.middleware import agent_hooks as hooks_module


@pytest.fixture
def db_guard(monkeypatch):
    """get_db_session 替换为计次桩：钩子若未正确分流会触库，直接暴露"""
    calls = {"count": 0}

    @asynccontextmanager
    async def _fake():
        calls["count"] += 1
        yield None

    monkeypatch.setattr(hooks_module, "get_db_session", _fake)
    return calls


async def _drain_pending_tasks():
    """等待钩子派发的后台任务完成，避免用例断言与任务回收竞态"""
    pending = list(hooks_module._pending_tasks)
    if pending:
        await __import__("asyncio").gather(*pending)


class TestSubagentSkipsLifecycleHooks:
    async def test_memory_extraction_skipped_for_subagent(self, monkeypatch, db_guard):
        """子图 run 不触发记忆提取"""
        touched = []

        async def _extract(*args, **kwargs):
            touched.append("extract")

        async def _save(*args, **kwargs):
            touched.append("save")

        monkeypatch.setattr(hooks_module, "extract_memories", _extract)
        monkeypatch.setattr(hooks_module, "save_extracted_memories", _save)

        state = {
            "is_subagent": True,
            "user_id": 10,
            "model_id": "m1",
            "conversation_id": 1,
            "messages": [{"role": "user", "content": "子任务描述"}],
        }
        assert await hooks_module._memory_extraction_hook(state) is None
        await _drain_pending_tasks()

        assert touched == []
        assert db_guard["count"] == 0

    async def test_title_update_skipped_for_subagent(self, monkeypatch, db_guard):
        """子图 run 不更新会话标题（不触库、不派发标题生成任务）"""
        state = {
            "is_subagent": True,
            "conversation_id": 1,
            "messages": [
                {"role": "user", "content": "子任务"},
                {"role": "assistant", "content": "完成"},
            ],
        }
        assert await hooks_module._title_update_hook(state) is None
        await _drain_pending_tasks()

        assert db_guard["count"] == 0


class TestMainRunLifecycleHooksUnchanged:
    async def test_memory_extraction_runs_for_main(self, monkeypatch, db_guard):
        """主图 run 记忆提取链路不变"""
        saved = []

        async def _extract(*args, **kwargs):
            return ["用户偏好简洁回答"]

        async def _save(user_id, memories):
            saved.append((user_id, memories))

        monkeypatch.setattr(hooks_module, "extract_memories", _extract)
        monkeypatch.setattr(hooks_module, "save_extracted_memories", _save)

        state = {
            "user_id": 10,
            "model_id": "m1",
            "conversation_id": 1,
            "messages": [{"role": "user", "content": "帮我总结"}],
        }
        assert await hooks_module._memory_extraction_hook(state) is None
        await _drain_pending_tasks()

        assert saved == [(10, ["用户偏好简洁回答"])]

    async def test_title_update_runs_for_main(self, monkeypatch, db_guard):
        """主图 run 自动标题链路不变（新对话触发 LLM 标题生成）"""
        monkeypatch.setattr(
            hooks_module,
            "ai_conversation_repository",
            SimpleNamespace(
                get_by_id=lambda db, cid: _async_value(SimpleNamespace(title="新对话", deleted=0))
            ),
        )
        generated = []

        async def _auto_generate_title(conversation_id, context_text):
            generated.append((conversation_id, context_text))

        import app.service.ai_conversation_service as conv_module

        monkeypatch.setattr(
            conv_module,
            "ai_conversation_service",
            SimpleNamespace(_auto_generate_title=_auto_generate_title),
        )

        state = {
            "conversation_id": 1,
            "messages": [
                {"role": "user", "content": "图片去雾"},
                {"role": "assistant", "content": "已完成"},
            ],
        }
        assert await hooks_module._title_update_hook(state) is None
        await _drain_pending_tasks()

        assert generated == [(1, "图片去雾 已完成")]


async def _async_value(value):
    return value
