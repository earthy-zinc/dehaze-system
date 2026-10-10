"""默认 Agent 模型跟随平台默认（需求 §2.11.12：默认 Agent 的模型=平台默认模型）。

平台默认切换（sys_dict）后，默认 Agent 行与已发布快照须随启动同步并重发布，
否则默认会话兜底与评测会继续用旧模型（本环境曾长期停在已失效的内置本地 qwen3-0.6b）。
"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from redis.asyncio import Redis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_agent_repository import AiAgentRepository
from app.repository.ai_agent_version_repository import AiAgentVersionRepository
from app.repository.ai_skill_repository import AiSkillRepository
from app.service import ai_agent_service as m
from app.service.ai_agent_service import AgentService
from app.service.ai_agent_version_service import AgentVersionService


def _agent(model_id: str):
    return SimpleNamespace(
        id=1, agent_code="default", name="默认助手", deleted=0, model_id=model_id
    )


def _published_ok():
    return SimpleNamespace(snapshot={"resolved_config": {"max_steps_react": 8}})


def _svc(agent, published, write_draft):
    return AgentService(
        # 替身：多方法结构型仓储桩（沿用 test_ai_agent_service 口径）
        ai_agent_repository=cast(
            AiAgentRepository, SimpleNamespace(get_by_code=AsyncMock(return_value=agent))
        ),
        ai_agent_version_repository=cast(
            AiAgentVersionRepository,
            SimpleNamespace(get_latest_published=AsyncMock(return_value=published)),
        ),
        ai_skill_repository=cast(AiSkillRepository, SimpleNamespace()),
        agent_version_service=cast(AgentVersionService, SimpleNamespace(_write_draft=write_draft)),
    )


async def test_model_synced_to_platform_default_and_republished(monkeypatch):
    agent = _agent("qwen3-0.6b")
    write = AsyncMock()
    monkeypatch.setattr(
        m, "resolve_platform_default_model", AsyncMock(return_value="Qwen/Qwen3-8B")
    )
    monkeypatch.setattr(m, "_defer_clear_agent_caches", lambda *a: None)

    redis = AsyncMock(spec=Redis)
    await _svc(agent, _published_ok(), write).ensure_default_agent(AsyncMock(), redis)

    assert agent.model_id == "Qwen/Qwen3-8B"
    write.assert_awaited_once()
    assert write.await_args is not None
    assert write.await_args.kwargs["status"] == 2


async def test_no_change_no_republish(monkeypatch):
    agent = _agent("Qwen/Qwen3-8B")
    write = AsyncMock()
    monkeypatch.setattr(
        m, "resolve_platform_default_model", AsyncMock(return_value="Qwen/Qwen3-8B")
    )
    monkeypatch.setattr(m, "_defer_clear_agent_caches", lambda *a: None)

    redis = AsyncMock(spec=Redis)
    await _svc(agent, _published_ok(), write).ensure_default_agent(AsyncMock(), redis)

    write.assert_not_awaited()


async def test_unresolvable_platform_default_skips_without_crash(monkeypatch):
    """平台默认未配置：记警告跳过同步，不抛异常、不覆写现有模型"""
    agent = _agent("qwen3-0.6b")
    write = AsyncMock()
    monkeypatch.setattr(
        m,
        "resolve_platform_default_model",
        AsyncMock(side_effect=BusinessException(ResultCode.BUSINESS_ERROR, "未配置平台默认模型")),
    )
    monkeypatch.setattr(m, "_defer_clear_agent_caches", lambda *a: None)

    redis = AsyncMock(spec=Redis)
    await _svc(agent, _published_ok(), write).ensure_default_agent(AsyncMock(), redis)

    assert agent.model_id == "qwen3-0.6b"
    write.assert_not_awaited()
