"""会话创建与后台标题生成的默认模型取值。

调用侧契约：新建会话未显式指定 model 时按用户偏好解析（偏好 > 平台默认）；
后台派生推理（标题生成）不读偏好，只取平台默认。
"""

from collections.abc import AsyncGenerator

import pytest

from app.infrastructure.llm.call.llm_client import LlmClient
from app.infrastructure.llm.common import LlmStreamChunk
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.schema.ai_conversation import ConversationCreate
from app.repository.ai_user_preference_repository import ai_user_preference_repository
from app.service.ai_conversation_service import AiConversationService
from tests.stubs.factories import seed_ai_model, set_platform_default_dict

pytestmark = pytest.mark.requires_db

_USER_ID = 8_800_004
_PLATFORM_DEFAULT = "platform-default-chat"


@pytest.fixture
async def platform_default(db, mock_redis) -> str:
    """把平台默认模型写进 sys_dict（真实字典行，事务回滚），与用户偏好构成偏好≠平台默认的场景"""
    await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
    return _PLATFORM_DEFAULT


async def _set_preference(db, model_id: str) -> None:
    await ai_user_preference_repository.upsert_default_model(
        db, _USER_ID, model_id, actor_id=_USER_ID
    )


class _StubLlmClient(LlmClient):
    """记录实际请求的模型标识并回放固定标题（真类型子类，构造注入）"""

    def __init__(self, title: str = "季度销售分析") -> None:
        self.models: list[str] = []
        self._title = title

    async def stream_chat(
        self,
        db,
        model_id: str,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        on_route_result=None,
        user_id: int | None = None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        self.models.append(model_id)
        yield LlmStreamChunk(type="text_delta", content=self._title)


class TestCreateConversation:
    async def test_preference_model_persisted(self, db, mock_redis, platform_default):
        await seed_ai_model(db, mock_redis, "conv-pref-chat")
        await _set_preference(db, "conv-pref-chat")
        result = await AiConversationService().create_conversation(
            db, _USER_ID, ConversationCreate(title="偏好模型")
        )
        assert result.model == "conv-pref-chat"
        conv = await db.get(SysAiConversation, result.id)
        assert conv.model == "conv-pref-chat"

    async def test_platform_default_persisted_without_preference(self, db, platform_default):
        result = await AiConversationService().create_conversation(
            db, _USER_ID, ConversationCreate(title="平台默认")
        )
        assert result.model == _PLATFORM_DEFAULT
        conv = await db.get(SysAiConversation, result.id)
        assert conv.model == _PLATFORM_DEFAULT

    async def test_explicit_model_wins_over_preference(self, db, mock_redis, platform_default):
        await seed_ai_model(db, mock_redis, "conv-pref-chat")
        await _set_preference(db, "conv-pref-chat")
        result = await AiConversationService().create_conversation(
            db, _USER_ID, ConversationCreate(title="显式指定", model="conv-explicit")
        )
        assert result.model == "conv-explicit"


class TestTitleGeneration:
    async def test_uses_platform_default_not_user_preference(
        self, db, mock_redis, platform_default
    ):
        await seed_ai_model(db, mock_redis, "conv-pref-chat")
        await seed_ai_model(db, mock_redis, _PLATFORM_DEFAULT)
        await _set_preference(db, "conv-pref-chat")
        created = await AiConversationService().create_conversation(
            db, _USER_ID, ConversationCreate(title="后台标题")
        )
        # 模拟后台派生推理场景：会话未锚定模型
        conv = await db.get(SysAiConversation, created.id)
        conv.model = None
        await db.flush()

        llm = _StubLlmClient()
        service = AiConversationService(llm_client=llm)
        await service._auto_generate_title(created.id, "帮我分析季度销售数据")

        assert llm.models == [_PLATFORM_DEFAULT]
        await db.refresh(conv)
        assert conv.title == "季度销售分析"
        assert conv.title_source == "auto"
