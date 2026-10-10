"""用户 AI 偏好服务：偏好默认对话模型的读写。

写入可见性规则与 `GET /ai/models/enabled?modelType=chat` 同源：模型不存在/已停用/
非对话型/超出用户会员等级一律 A0601 且不落库；每用户一行，重复写覆盖值、清除写 NULL。
"""

import pytest
from sqlalchemy import select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_user_preference import SysAiUserPreference
from app.service.ai_user_preference_service import ai_user_preference_service
from tests.stubs.factories import seed_ai_model

pytestmark = pytest.mark.requires_db

_USER_ID = 8_800_001


async def _preference_row_count(db, user_id: int) -> int:
    stmt = select(SysAiUserPreference.id).where(SysAiUserPreference.user_id == user_id)
    return len((await db.execute(stmt)).scalars().all())


class TestReadWrite:
    async def test_missing_record_reads_none(self, db):
        assert await ai_user_preference_service.get_default_model(db, _USER_ID) is None

    async def test_set_chat_model_persists_single_row(self, db, mock_redis):
        await seed_ai_model(db, mock_redis, "pref-chat-a")
        await ai_user_preference_service.set_default_model(db, mock_redis, _USER_ID, "pref-chat-a")
        assert await ai_user_preference_service.get_default_model(db, _USER_ID) == "pref-chat-a"
        assert await _preference_row_count(db, _USER_ID) == 1

    async def test_repeated_set_overwrites_without_new_row(self, db, mock_redis):
        await seed_ai_model(db, mock_redis, "pref-chat-a")
        await seed_ai_model(db, mock_redis, "pref-chat-b")
        await ai_user_preference_service.set_default_model(db, mock_redis, _USER_ID, "pref-chat-a")
        await ai_user_preference_service.set_default_model(db, mock_redis, _USER_ID, "pref-chat-b")
        assert await ai_user_preference_service.get_default_model(db, _USER_ID) == "pref-chat-b"
        assert await _preference_row_count(db, _USER_ID) == 1

    async def test_clear_keeps_row_with_null_value(self, db, mock_redis):
        await seed_ai_model(db, mock_redis, "pref-chat-a")
        await ai_user_preference_service.set_default_model(db, mock_redis, _USER_ID, "pref-chat-a")
        await ai_user_preference_service.set_default_model(db, mock_redis, _USER_ID, None)
        assert await ai_user_preference_service.get_default_model(db, _USER_ID) is None
        row = (
            await db.execute(
                select(SysAiUserPreference).where(SysAiUserPreference.user_id == _USER_ID)
            )
        ).scalar_one()
        assert row.default_model is None


class TestRejectUnavailableModel:
    """四类不可见模型均 A0601，且偏好表不落任何行"""

    async def test_unknown_model_rejected(self, db, mock_redis):
        with pytest.raises(BusinessException) as exc:
            await ai_user_preference_service.set_default_model(
                db, mock_redis, _USER_ID, "pref-ghost"
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE
        assert await _preference_row_count(db, _USER_ID) == 0

    async def test_disabled_model_rejected(self, db, mock_redis):
        await seed_ai_model(db, mock_redis, "pref-disabled", status=0)
        with pytest.raises(BusinessException) as exc:
            await ai_user_preference_service.set_default_model(
                db, mock_redis, _USER_ID, "pref-disabled"
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE
        assert await _preference_row_count(db, _USER_ID) == 0

    async def test_non_chat_model_rejected(self, db, mock_redis):
        await seed_ai_model(db, mock_redis, "pref-embedding", model_type="embedding")
        with pytest.raises(BusinessException) as exc:
            await ai_user_preference_service.set_default_model(
                db, mock_redis, _USER_ID, "pref-embedding"
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE
        assert await _preference_row_count(db, _USER_ID) == 0

    async def test_model_above_user_level_rejected(self, db, mock_redis):
        """无会员记录的用户等级为 0，vip_level=1 的模型不可见"""
        await seed_ai_model(db, mock_redis, "pref-vip1", vip_level=1)
        with pytest.raises(BusinessException) as exc:
            await ai_user_preference_service.set_default_model(
                db, mock_redis, _USER_ID, "pref-vip1"
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE
        assert await _preference_row_count(db, _USER_ID) == 0
