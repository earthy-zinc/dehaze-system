"""语音引擎管理服务 VoiceAdminService 单元测试（业务边界）。

覆盖易忽略的管理约束：
- is_default 每能力维度唯一（新建同 engine_type 的 default 引擎清除旧 default）
- provider_code 删除后不可复用（软删记录仍占用联合唯一键）
- 删除引擎时存在启用模型引用 → 拒绝（DATA_BIND_EXISTS）
"""

import pytest
from fakeredis import FakeAsyncRedis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.voice.provider.registry import VoiceEngineRegistry, _ENGINE_CACHE_KEY
from app.models.schema.voice_admin import (
    VoiceModelCreate,
    VoiceProviderCreate,
    VoiceProviderUpdate,
)
from app.repository.voice_provider_repository import voice_provider_repository
from app.service.voice.voice_admin_service import VoiceAdminService, voice_admin_service


def _provider_form(code: str, engine_type: str, name: str, is_default: int = 0):
    return VoiceProviderCreate(
        provider_code=code, engine_type=engine_type, display_name=name, is_default=is_default
    )


@pytest.mark.requires_db
async def test_set_default_provider_clears_other_default(db):
    """is_default 每能力唯一：新建同 engine_type 的 default 引擎清除旧 default"""
    first = await voice_admin_service.create_provider(
        db, _provider_form("local", "asr", "本地", is_default=1)
    )
    second = await voice_admin_service.create_provider(
        db, _provider_form("aliyun", "asr", "阿里云", is_default=1)
    )

    assert second.is_default == 1
    first_re = await voice_provider_repository.get_by_id(db, first.id)
    assert first_re.is_default == 0  # 旧 default 已被清除，保证每能力仅一条 default


@pytest.mark.requires_db
async def test_provider_code_not_reusable_after_delete(db):
    """provider_code 删除后不可复用：软删记录仍占用 (provider_code, engine_type)"""
    p = await voice_admin_service.create_provider(
        db, _provider_form("tencent", "asr", "腾讯")
    )
    await voice_admin_service.delete_provider(db, p.id)

    with pytest.raises(BusinessException) as exc:
        await voice_admin_service.create_provider(
            db, _provider_form("tencent", "asr", "腾讯2")
        )
    assert exc.value.code == ResultCode.DATA_EXISTS


@pytest.mark.requires_db
async def test_delete_provider_with_enabled_model_rejected(db):
    """删除引擎时存在启用模型引用 → 拒绝（DATA_BIND_EXISTS）"""
    p = await voice_admin_service.create_provider(
        db, _provider_form("xfyun", "asr", "讯飞")
    )
    await voice_admin_service.create_model(
        db,
        VoiceModelCreate(
            provider_id=p.id, model_id="sensevoice", engine_type="asr",
            model_type="stream", display_name="流式",
        ),
    )

    with pytest.raises(BusinessException) as exc:
        await voice_admin_service.delete_provider(db, p.id)
    assert exc.value.code == ResultCode.DATA_BIND_EXISTS


@pytest.mark.requires_db
async def test_default_change_invalidates_engine_cache(db):
    """is_default/status 变更失效默认引擎 Redis 缓存（后端实现 §2.4 切换即时生效）；无关变更不失效"""
    redis = FakeAsyncRedis(decode_responses=True)

    async def _redis_factory():
        return redis

    spy_registry = VoiceEngineRegistry(
        repository=voice_provider_repository,
        session_factory=None,  # 本测试不触发 Provider 路由，session_factory 不会被调用
        engine_available=lambda t: True,
        redis_factory=_redis_factory,
    )
    service = VoiceAdminService(engine_registry=spy_registry)
    p = await service.create_provider(db, _provider_form("spark", "asr", "星火", is_default=1))
    # 新建默认引擎：缓存被失效（key 不残留）
    assert await redis.get(_ENGINE_CACHE_KEY.format("asr")) is None

    # 无关字段（remark）变更：不失效
    await redis.set(_ENGINE_CACHE_KEY.format("asr"), "cached")
    await service.update_provider(db, p.id, VoiceProviderUpdate(remark="备注"))
    assert await redis.get(_ENGINE_CACHE_KEY.format("asr")) == "cached"

    # status 变更：失效
    await service.update_provider(db, p.id, VoiceProviderUpdate(status=0))
    assert await redis.get(_ENGINE_CACHE_KEY.format("asr")) is None

    # 删除引擎：失效
    await redis.set(_ENGINE_CACHE_KEY.format("asr"), "cached")
    await service.delete_provider(db, p.id)
    assert await redis.get(_ENGINE_CACHE_KEY.format("asr")) is None


@pytest.mark.requires_db
async def test_list_models_filters_by_engine_type(db):
    """模型/音色列表按 engine_type 筛选，含全部状态（管理端展示）"""
    p = await voice_admin_service.create_provider(
        db, _provider_form("local", "asr", "本地ASR")
    )
    await voice_admin_service.create_model(
        db,
        VoiceModelCreate(
            provider_id=p.id, model_id="sensevoice", engine_type="asr",
            model_type="stream", display_name="流式", status=1,
        ),
    )
    await voice_admin_service.create_model(
        db,
        VoiceModelCreate(
            provider_id=p.id, model_id="paraformer", engine_type="asr",
            model_type="offline", display_name="离线", status=0,
        ),
    )

    asr_models = await voice_admin_service.list_models(db, engine_type="asr")
    tts_models = await voice_admin_service.list_models(db, engine_type="tts")
    all_models = await voice_admin_service.list_models(db)

    assert [m.model_id for m in asr_models] == ["paraformer", "sensevoice"]  # 含禁用模型，按 model_id 排序
    assert tts_models == []
    assert len(all_models) == 2


@pytest.mark.requires_db
async def test_test_connection_local_reports_connected(db):
    """local 引擎走进程内引擎，连通性测试直接报告已连接"""
    p = await voice_admin_service.create_provider(
        db, _provider_form("local", "tts", "本地Piper", is_default=1)
    )

    result = await voice_admin_service.test_connection(db, p.id)

    assert result == {"result": "本地引擎", "connected": True}


@pytest.mark.requires_db
async def test_test_connection_cloud_skips_until_vendor_protocol_ready(db):
    """云端厂商协议未接入：显式跳过连通性测试（connected=None，仅提示不阻断）"""
    p = await voice_admin_service.create_provider(
        db, _provider_form("azure", "tts", "Azure TTS")
    )

    result = await voice_admin_service.test_connection(db, p.id)

    assert result["connected"] is None
    assert "未接入" in result["result"]
