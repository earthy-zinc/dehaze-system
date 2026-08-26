import pytest

from app.service.ai.service import memory_es_service

pytestmark = pytest.mark.requires_db
from app.service.ai.service.memory_es_service import (
    DEFAULT_DIMS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
)
from app.service.ai.service import memory_extraction
from tests.stubs.fakes import LLMChunk

_MESSAGE_CONTENT = (
    "请帮我处理一下这张图片，重点是把雾霾去掉并保留边缘细节，"
    "同时把天空和地面分开处理，最后输出一张对比效果图"
)


def _patch_extraction(monkeypatch, payload: str):
    async def fake_stream(db, model_id, messages, **kwargs):
        yield LLMChunk(type="text_delta", content=payload)

    async def fake_score(db, model_id, content):
        return 50

    async def no_existing(db, user_id, limit=50):
        return []

    monkeypatch.setattr(memory_extraction.llm_client, "stream_chat", fake_stream)
    monkeypatch.setattr(memory_extraction, "_score_importance", fake_score)
    monkeypatch.setattr(
        memory_extraction.ai_memory_repository, "get_active_by_user", no_existing
    )


async def _extract(db, monkeypatch, payload: str) -> list[dict]:
    _patch_extraction(monkeypatch, payload)
    return await memory_extraction.extract_memories(
        user_id=1,
        model_id="m1",
        messages=[{"role": "user", "content": _MESSAGE_CONTENT}],
    )


class TestEmbeddingConfig:

    async def test_load_embedding_config_from_dict(self, db):
        # 直连真实测试库：config/sql 种子已含 ai_embedding 三键，断言真实查询结果。
        # sys_dict.value 为字符串列，故 dims/model/provider_code 均为字符串。
        config = await memory_es_service._load_embedding_config()
        assert config["provider_code"] == "openai"
        assert config["model"] == "text-embedding-3-small"
        assert config["dims"] == "1536"

    async def test_embedding_config_fallback_on_error(self, monkeypatch):
        # 仓储故障注入：dict 查询抛错时回落内置默认配置（非种子值）
        async def _boom(db, type_code):
            raise RuntimeError("db down")

        monkeypatch.setattr(
            memory_es_service.dict_repository, "list_enabled_by_type_code", _boom
        )
        config = await memory_es_service._load_embedding_config()
        assert config["provider_code"] == DEFAULT_PROVIDER
        assert config["model"] == DEFAULT_MODEL
        assert config["dims"] == DEFAULT_DIMS


class TestExtractMemoriesParsing:

    async def test_semantic_metadata_and_type_mapping(self, db, monkeypatch):
        payload = '[{"type":"semantic","content":"用户偏好简洁回复",' \
                  '"metadata":{"category":"preference","is_preference":true}}]'
        out = await _extract(db, monkeypatch, payload)
        assert len(out) == 1
        assert out[0]["memory_type"] == "semantic"
        assert out[0]["metadata"]["is_preference"] is True
        assert out[0]["content"] == "用户偏好简洁回复"
        assert out[0]["importance"] == 50

    async def test_procedural_metadata(self, db, monkeypatch):
        payload = '[{"type":"procedural","content":"先去雾再评估",' \
                  '"metadata":{"skill":"dehaze","steps":"先处理再评估"}}]'
        out = await _extract(db, monkeypatch, payload)
        assert out[0]["memory_type"] == "procedural"
        assert out[0]["metadata"]["skill"] == "dehaze"

    async def test_episodic_metadata(self, db, monkeypatch):
        payload = '[{"type":"episodic","content":"上周处理雾图",' \
                  '"metadata":{"event":"处理雾图","outcome":"满意"}}]'
        out = await _extract(db, monkeypatch, payload)
        assert out[0]["metadata"]["event"] == "处理雾图"

    async def test_empty_list(self, db, monkeypatch):
        out = await _extract(db, monkeypatch, "[]")
        assert out == []

    async def test_invalid_item_skipped_and_type_defaulted(self, db, monkeypatch):
        payload = '[{"type":"semantic","content":"有效"}, "not-a-dict", {}, {"content":"缺类型"}]'
        out = await _extract(db, monkeypatch, payload)
        assert len(out) == 2
        assert out[0]["memory_type"] == "semantic"
        assert out[0]["content"] == "有效"
        assert out[1]["memory_type"] == "semantic"

    async def test_pii_content_masked_in_output(self, db, monkeypatch):
        payload = '[{"type":"semantic","content":"手机：13800138000，请尽快联系我",' \
                  '"metadata":{}}]'
        out = await _extract(db, monkeypatch, payload)
        assert out[0]["content"] == "手机：***，请尽快联系我"


class TestSaveMetadata:

    def test_metadata_passed_to_entity(self):
        from app.models.entity.sys_ai_memory import SysAiMemory

        assert "metadata" in SysAiMemory.__table__.columns.keys()
        assert "delete_time" in SysAiMemory.__table__.columns.keys()
