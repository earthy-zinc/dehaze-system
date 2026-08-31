"""会话类似问题推荐开关（suggestions_enabled）单测：创建/更新读写与 suggestion 消费"""

import pytest

pytestmark = pytest.mark.requires_db

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.schema.ai_conversation import ConversationCreate, ConversationUpdate
from app.service.ai.service.suggestion_service import suggestion_service
from app.service.ai_conversation_service import AiConversationService

_USER_ID = 1


def _service():
    return AiConversationService()


class TestCreateToggle:
    async def test_default_enabled(self, db):
        result = await _service().create_conversation(
            db, _USER_ID, ConversationCreate(title="默认开启")
        )
        assert result.suggestions_enabled == 1

    async def test_create_disabled(self, db):
        result = await _service().create_conversation(
            db, _USER_ID, ConversationCreate(title="关闭推荐", suggestionsEnabled=False)
        )
        assert result.suggestions_enabled == 0

    async def test_persists_to_db(self, db):
        result = await _service().create_conversation(
            db, _USER_ID, ConversationCreate(suggestionsEnabled=False)
        )
        conv = await db.get(SysAiConversation, result.id)
        assert conv.suggestions_enabled == 0


class TestUpdateToggle:
    async def test_toggle_off(self, db):
        created = await _service().create_conversation(
            db, _USER_ID, ConversationCreate()
        )
        result = await _service().update_conversation(
            db, created.id, _USER_ID, ConversationUpdate(suggestionsEnabled=False)
        )
        assert result.suggestions_enabled == 0

    async def test_toggle_on(self, db):
        created = await _service().create_conversation(
            db, _USER_ID, ConversationCreate(suggestionsEnabled=False)
        )
        result = await _service().update_conversation(
            db, created.id, _USER_ID, ConversationUpdate(suggestionsEnabled=True)
        )
        assert result.suggestions_enabled == 1


class TestSuggestionConsumption:
    async def test_disabled_conversation_skips(self, db):
        """开关关闭的会话不应推送 suggestions（_is_enabled 是推送前最后一道闸）"""
        created = await _service().create_conversation(
            db, _USER_ID, ConversationCreate(suggestionsEnabled=False)
        )
        conv = await db.get(SysAiConversation, created.id)
        assert await suggestion_service._is_enabled(conv) is False

    async def test_enabled_conversation_passes(self, db):
        created = await _service().create_conversation(db, _USER_ID, ConversationCreate())
        conv = await db.get(SysAiConversation, created.id)
        assert await suggestion_service._is_enabled(conv) is True
