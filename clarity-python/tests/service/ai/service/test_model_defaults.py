"""默认对话模型解析链与输入预算校验。

解析链：用户偏好默认模型（存在 + 启用 + model_type=chat）> Agent 配置模型
（会话锚定 Agent 的已发布版本快照 model_id，无已发布版本回退主表）> 平台默认
（sys_dict: ai_model.default_model，status=0 视同缺项）；皆无 → A0500，
不回落代码常量。不传 user_id/agent_code（后台派生推理）时跳过对应层，直取平台默认。
平台默认按库内实际状态构造（真实 sys_dict 行，事务回滚），不假设种子行天然存在/缺失。
"""

from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_user_preference_repository import ai_user_preference_repository
from app.service.ai.service import model_defaults
from app.service.ai.service.model_defaults import (
    assert_message_fits_context,
    assert_message_fits_model,
    input_budget_tokens,
    resolve_default_chat_model,
)
from tests.stubs.factories import seed_ai_model, set_platform_default_dict

pytestmark = pytest.mark.requires_db

_USER_ID = 8_800_002
_PLATFORM_DEFAULT = "platform-default-chat"


def test_platform_default_dict_contract():
    assert (model_defaults.DEFAULT_MODEL_DICT_TYPE, model_defaults.DEFAULT_MODEL_DICT_NAME) == (
        "ai_model",
        "default_model",
    )


def test_default_model_dict_key_contract():
    """平台默认模型字典键口径：chat 用 `default_model`，其余按类型增键

    与 config/sql/data/sys_dict.sql 的键一致；消费方（/ai/models/enabled 的 isDefault
    标记、语音/图像路由）据此按 model_type 取键，键写错即静默取不到默认模型。
    """
    assert model_defaults.default_model_dict_key("chat") == "default_model"
    assert model_defaults.default_model_dict_key("image_edit") == "default_model.image_edit"
    assert model_defaults.default_model_dict_key("asr") == "default_model.asr"
    assert model_defaults.default_model_dict_key("tts") == "default_model.tts"


async def _set_preference(db, user_id: int, model_id: str | None) -> None:
    await ai_user_preference_repository.upsert_default_model(
        db, user_id, model_id, actor_id=user_id
    )


class TestResolveChain:
    async def test_without_user_id_uses_platform_default(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        assert await resolve_default_chat_model(db) == _PLATFORM_DEFAULT

    async def test_user_preference_takes_priority(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(db, mock_redis, "pref-chat")
        await _set_preference(db, _USER_ID, "pref-chat")
        assert await resolve_default_chat_model(db, _USER_ID) == "pref-chat"

    async def test_disabled_preference_falls_back(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(db, mock_redis, "pref-disabled", status=0)
        await _set_preference(db, _USER_ID, "pref-disabled")
        assert await resolve_default_chat_model(db, _USER_ID) == _PLATFORM_DEFAULT

    async def test_soft_deleted_preference_falls_back(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(db, mock_redis, "pref-deleted")
        await _set_preference(db, _USER_ID, "pref-deleted")
        model = await ai_model_repository.get_by_model_id(db, "pref-deleted")
        assert model is not None
        model.deleted = model.id
        await db.flush()
        assert await resolve_default_chat_model(db, _USER_ID) == _PLATFORM_DEFAULT

    async def test_non_chat_preference_falls_back(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(db, mock_redis, "pref-embedding", model_type="embedding")
        await _set_preference(db, _USER_ID, "pref-embedding")
        assert await resolve_default_chat_model(db, _USER_ID) == _PLATFORM_DEFAULT

    async def test_empty_preference_value_falls_back(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await _set_preference(db, _USER_ID, None)
        assert await resolve_default_chat_model(db, _USER_ID) == _PLATFORM_DEFAULT

    async def test_preference_survives_missing_platform_default(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, None)
        await seed_ai_model(db, mock_redis, "pref-chat")
        await _set_preference(db, _USER_ID, "pref-chat")
        assert await resolve_default_chat_model(db, _USER_ID) == "pref-chat"

    async def test_both_unconfigured_raises_a0500(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, None)
        with pytest.raises(BusinessException) as exc:
            await resolve_default_chat_model(db, _USER_ID)
        assert exc.value.code == ResultCode.BUSINESS_ERROR
        assert "未配置平台默认模型" in exc.value.message

    async def test_disabled_dict_item_treated_as_missing(self, db, mock_redis):
        """status=0 的字典项不生效（视同缺项），不得返回停用值"""
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT, status=0)
        with pytest.raises(BusinessException) as exc:
            await resolve_default_chat_model(db)
        assert exc.value.code == ResultCode.BUSINESS_ERROR
        assert "未配置平台默认模型" in exc.value.message

    async def test_blank_dict_value_treated_as_missing(self, db, mock_redis):
        """字典值为空串等同未配置 → A0500（不得把空串当模型标识返回）"""
        await set_platform_default_dict(db, mock_redis, "")
        with pytest.raises(BusinessException) as exc:
            await resolve_default_chat_model(db)
        assert exc.value.code == ResultCode.BUSINESS_ERROR
        assert "未配置平台默认模型" in exc.value.message


class TestAgentLevel:
    """Agent 级解析：用户偏好 > Agent 配置模型（已发布快照优先，回退主表）> 平台默认"""

    @staticmethod
    async def _seed_agent(
        db, code: str, main_model: str, *, published_model: str | None = None
    ) -> None:
        agent = SysAiAgent(agent_code=code, name=f"A-{code}", model_id=main_model)
        db.add(agent)
        await db.flush()
        if published_model is not None:
            db.add(
                SysAiAgentVersion(
                    agent_id=agent.id,
                    version_no=1,
                    snapshot={"model_id": published_model},
                    status=2,
                )
            )
            await db.flush()

    async def test_agent_published_model_used_without_preference(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await self._seed_agent(
            db, "t-agent-pub", "agent-main-model", published_model="agent-pub-model"
        )
        assert await resolve_default_chat_model(db, _USER_ID, "t-agent-pub") == "agent-pub-model"

    async def test_user_preference_beats_agent_model(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(db, mock_redis, "pref-chat")
        await _set_preference(db, _USER_ID, "pref-chat")
        await self._seed_agent(db, "t-agent-pref", "agent-model", published_model="agent-pub")
        assert await resolve_default_chat_model(db, _USER_ID, "t-agent-pref") == "pref-chat"

    async def test_agent_without_published_version_uses_main_table(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await self._seed_agent(db, "t-agent-draft", "agent-main-only")
        assert await resolve_default_chat_model(db, _USER_ID, "t-agent-draft") == "agent-main-only"

    async def test_unknown_agent_falls_back_to_platform_default(self, db, mock_redis):
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        assert await resolve_default_chat_model(db, _USER_ID, "no-such-agent") == _PLATFORM_DEFAULT
        assert await resolve_default_chat_model(db, _USER_ID) == _PLATFORM_DEFAULT

    async def test_agent_level_still_needs_platform_default_when_agent_absent(self, db, mock_redis):
        """无 Agent 上下文时行为不变：平台默认缺失仍报 A0500（不因新增层级被吞掉）"""
        await set_platform_default_dict(db, mock_redis, None)
        with pytest.raises(BusinessException) as exc:
            await resolve_default_chat_model(db, _USER_ID, "no-such-agent")
        assert exc.value.code == ResultCode.BUSINESS_ERROR


class TestInputBudget:
    def test_default_output_reserve_subtracted(self):
        model = SimpleNamespace(max_context_tokens=8000, max_output_tokens=None)
        assert input_budget_tokens(model) == 8000 - 1024 - 200

    def test_floor_is_512(self):
        model = SimpleNamespace(max_context_tokens=600, max_output_tokens=4096)
        assert input_budget_tokens(model) == 512


class TestMessageBudget:
    async def test_oversized_message_rejected(self, db, mock_redis):
        await seed_ai_model(
            db, mock_redis, "budget-chat", max_context_tokens=2048, max_output_tokens=1024
        )
        with pytest.raises(BusinessException) as exc:
            await assert_message_fits_context(db, "x" * 900, "budget-chat")
        assert exc.value.code == ResultCode.BUSINESS_ERROR
        assert exc.value.message == "单条消息超出模型上下文预算（824 字符，模型 budget-chat）"

    async def test_message_within_budget_passes(self, db, mock_redis):
        await seed_ai_model(
            db, mock_redis, "budget-chat", max_context_tokens=2048, max_output_tokens=1024
        )
        await assert_message_fits_context(db, "x" * 800, "budget-chat")

    async def test_unknown_model_skips_check(self, db):
        await assert_message_fits_context(db, "x" * 10000, "no-such-model")

    async def test_unresolved_model_follows_preference_budget(self, db, mock_redis):
        """未显式指定模型时按解析链取预算：偏好模型预算小 → 超出即报错（不是平台默认的大预算）"""
        await set_platform_default_dict(db, mock_redis, _PLATFORM_DEFAULT)
        await seed_ai_model(
            db, mock_redis, _PLATFORM_DEFAULT, max_context_tokens=8000, max_output_tokens=1024
        )
        await seed_ai_model(
            db, mock_redis, "budget-small", max_context_tokens=2048, max_output_tokens=1024
        )
        await _set_preference(db, _USER_ID, "budget-small")
        with pytest.raises(BusinessException) as exc:
            await assert_message_fits_context(db, "x" * 900, None, user_id=_USER_ID)
        assert "budget-small" in exc.value.message

    async def test_loaded_model_row_check_shares_error_message(self, db, mock_redis):
        await seed_ai_model(
            db, mock_redis, "budget-chat", max_context_tokens=2048, max_output_tokens=1024
        )
        model = await ai_model_repository.get_by_model_id(db, "budget-chat")
        assert model is not None
        with pytest.raises(BusinessException) as exc:
            assert_message_fits_model("x" * 900, model, "budget-chat")
        assert exc.value.message == "单条消息超出模型上下文预算（824 字符，模型 budget-chat）"
