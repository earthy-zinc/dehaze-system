from app.service.ai import memory_extraction
from app.service.ai import memory_es_service
from tests.stubs import LLMChunk, NullDBSession

_MESSAGE_CONTENT = (
    "请帮我处理一下这张图片，重点是把雾霾去掉并保留边缘细节，"
    "同时把天空和地面分开处理，最后输出一张对比效果图"
)


def _patch_extraction(monkeypatch, payload: str):
    async def fake_stream(db, redis, model_id, messages, **kwargs):
        yield LLMChunk(type="text_delta", content=payload)

    async def fake_score(db, model_id, content):
        return 50

    async def no_existing(db, user_id, limit=50):
        return []

    monkeypatch.setattr(memory_extraction.llm_client, "stream_chat", fake_stream)
    monkeypatch.setattr(memory_extraction, "_score_importance", fake_score)
    monkeypatch.setattr(memory_extraction, "get_db_session", NullDBSession)
    monkeypatch.setattr(
        memory_extraction.ai_memory_repository, "get_active_by_user", no_existing
    )


async def _extract(monkeypatch, payload: str) -> list[dict]:
    _patch_extraction(monkeypatch, payload)
    return await memory_extraction.extract_memories(
        user_id=1,
        model_id="m1",
        messages=[{"role": "user", "content": _MESSAGE_CONTENT}],
    )


class TestEmbeddingConfig:

    async def test_load_embedding_config_from_dict(self, monkeypatch):
        row = type("D", (), {"name": "provider_code", "value": "qwen", "status": 1})

        class FakeResult:
            def scalars(self):
                return self

            def all(self):
                return [row]

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def execute(self, stmt):
                return FakeResult()

        monkeypatch.setattr("app.database.get_db_session", FakeSession)
        config = await memory_es_service._load_embedding_config()
        assert config["provider_code"] == "qwen"
        assert config["model"] == "text-embedding-3-small"
        assert config["dims"] == 1536

    async def test_embedding_config_fallback_on_error(self, monkeypatch):
        class BoomSession:
            async def __aenter__(self):
                raise RuntimeError("db down")

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr("app.database.get_db_session", BoomSession)
        config = await memory_es_service._load_embedding_config()
        assert config["provider_code"] == "openai"
        assert config["dims"] == 1536


class TestExtractMemoriesParsing:

    async def test_semantic_metadata_and_type_mapping(self, monkeypatch):
        payload = '[{"type":"semantic","content":"用户偏好简洁回复",' \
                  '"metadata":{"category":"preference","is_preference":true}}]'
        out = await _extract(monkeypatch, payload)
        assert len(out) == 1
        assert out[0]["memory_type"] == "semantic"
        assert out[0]["metadata"]["is_preference"] is True
        assert out[0]["content"] == "用户偏好简洁回复"
        assert out[0]["importance"] == 50

    async def test_procedural_metadata(self, monkeypatch):
        payload = '[{"type":"procedural","content":"先去雾再评估",' \
                  '"metadata":{"skill":"dehaze","steps":"先处理再评估"}}]'
        out = await _extract(monkeypatch, payload)
        assert out[0]["memory_type"] == "procedural"
        assert out[0]["metadata"]["skill"] == "dehaze"

    async def test_episodic_metadata(self, monkeypatch):
        payload = '[{"type":"episodic","content":"上周处理雾图",' \
                  '"metadata":{"event":"处理雾图","outcome":"满意"}}]'
        out = await _extract(monkeypatch, payload)
        assert out[0]["metadata"]["event"] == "处理雾图"

    async def test_empty_list(self, monkeypatch):
        out = await _extract(monkeypatch, "[]")
        assert out == []

    async def test_invalid_item_skipped_and_type_defaulted(self, monkeypatch):
        payload = '[{"type":"semantic","content":"有效"}, "not-a-dict", {}, {"content":"缺类型"}]'
        out = await _extract(monkeypatch, payload)
        assert len(out) == 2
        assert out[0]["memory_type"] == "semantic"
        assert out[0]["content"] == "有效"
        assert out[1]["memory_type"] == "semantic"

    async def test_pii_content_masked_in_output(self, monkeypatch):
        payload = '[{"type":"semantic","content":"手机：13800138000，请尽快联系我",' \
                  '"metadata":{}}]'
        out = await _extract(monkeypatch, payload)
        assert out[0]["content"] == "手机：***，请尽快联系我"


class TestSaveMetadata:

    def test_metadata_passed_to_entity(self):
        from app.models.entity.sys_ai_memory import SysAiMemory

        assert "metadata" in SysAiMemory.__table__.columns.keys()
        assert "delete_time" in SysAiMemory.__table__.columns.keys()
