from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_agent_repository import AiAgentRepository
from app.service.ai_message_service import AiMessageService

# 测试替身：仓储层已 mock，db 仅传参占位
_DB: AsyncSession = AsyncMock(spec=AsyncSession)


def _conv(agent_code="default", model_config=None):
    return SimpleNamespace(agent_code=agent_code, model_config=model_config or {})


def _agent(reasoning_mode="auto"):
    return SimpleNamespace(reasoning_mode=reasoning_mode)


class _AgentRepo(AiAgentRepository):
    """测试替身：仅实现 get_by_code（返回注入的 agent）。"""

    def __init__(self, agent):
        self._agent = agent

    async def get_by_code(self, *args, **kwargs):
        return self._agent


class TestNeedsToolCall:
    def _service(self, agent):
        return AiMessageService(ai_agent_repository=_AgentRepo(agent))

    async def test_defaults_true_when_no_config(self):
        assert await self._service(None)._needs_tool_call(_DB, _conv()) is True

    async def test_explicit_need_tools_false_wins(self):
        conv = _conv(model_config={"needTools": False})
        assert await self._service(_agent("react"))._needs_tool_call(_DB, conv) is False

    async def test_direct_agent_skips_validation(self):
        assert await self._service(_agent("direct"))._needs_tool_call(_DB, _conv()) is False

    async def test_non_direct_agent_defaults_true(self):
        assert await self._service(_agent("auto"))._needs_tool_call(_DB, _conv()) is True
