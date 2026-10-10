"""数据工厂：纯数据构造函数（redis/orm/conv/member/benefit/context 等）。

本模块只放"构造测试数据"的纯函数（factories），不含协议仿真或边界 patch。
协议仿真见 tests.stubs.fakes，patch 辅助见 tests.stubs.mocks。
"""

from datetime import datetime
from types import SimpleNamespace

from fakeredis import FakeAsyncRedis

from app.models.schema.ai_conversation import AiModelType


async def fake_redis(data: dict | None = None) -> FakeAsyncRedis:
    """构造带初始数据的 fakeredis 客户端（异步工厂，测试内 await 使用）"""
    client = FakeAsyncRedis(decode_responses=True)
    for key, value in (data or {}).items():
        await client.set(key, value)
    return client


def make_orm_mem(
    id_,
    type_,
    content,
    importance=50,
    metadata=None,
    last_accessed=None,
    create=None,
    status=1,
    archived=0,
    deleted=0,
    **extra,
):
    fields = {
        "id": id_,
        "memory_type": type_,
        "content": content,
        "importance": importance,
        "metadata_": metadata,
        "last_accessed_at": last_accessed,
        "create_time": create or datetime.now(),
        "status": status,
        "archived": archived,
        "deleted": deleted,
    }
    fields.update(extra)
    return type("M", (), fields)()


def make_conv(**overrides):
    fields = {
        "id": 1,
        "user_id": 1,
        "system_prompt": None,
        "current_branch_message_id": None,
        "summary": None,
        "agent_code": None,
        "agent_version": None,
        "status": 1,
        "model": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_member(level_code="level_0"):
    return SimpleNamespace(level_code=level_code)


def make_benefit(multimodal_limit=0, **extra):
    fields = {"multimodal_limit": multimodal_limit}
    fields.update(extra)
    return SimpleNamespace(**fields)


def repo_returns(obj):
    """get_by_id 恒返回固定对象的仓储桩（ai_message_repository 等）。"""

    class _Repo:
        async def get_by_id(self, db, msg_id):
            return obj

    return _Repo()


def make_user_context(id=1, username="u", **overrides):
    from app.dependencies.auth import UserContext

    fields = {"id": id, "username": username}
    fields.update(overrides)
    return UserContext(**fields)


async def seed_ai_model(
    db,
    redis,
    model_id: str,
    *,
    model_type: AiModelType = "chat",
    status: int = 1,
    vip_level: int = 0,
    max_context_tokens: int = 8192,
    max_output_tokens: int = 4096,
    provider_id: int = 999,
):
    """播种真实 sys_ai_model 行（经 create_model，同时失效模型列表缓存）。

    默认档位：启用 / 等级 0 / chat，即所有用户可见的普通对话模型。
    """
    from app.models.schema.ai_conversation import AiModelCreate
    from app.service.ai_model_service import ai_model_service

    form = AiModelCreate(
        provider_id=provider_id,
        model_id=model_id,
        model_type=model_type,
        dimension=1024 if model_type == "embedding" else None,
        display_name=f"M-{model_id}",
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        status=status,
        vip_level=vip_level,
    )
    return await ai_model_service.create_model(db, redis, form)


async def set_platform_default_dict(db, redis, value: str | None, *, status: int = 1) -> None:
    """构造平台默认模型的字典行（sys_dict: ai_model/default_model）。

    value=None 表示字典项缺项（软删现有行）；写入后清 `dict:value:` 读缓存，
    避免同一用例内先读后改仍命中旧值（字典读缓存 TTL 1 小时）。事务由 db fixture 回滚。
    """
    from app.infrastructure.cache.cache import CacheService
    from app.models.entity.sys_dict import SysDict
    from app.repository.dict_repository import dict_repository
    from app.service.ai.service.model_defaults import (
        DEFAULT_MODEL_DICT_NAME,
        DEFAULT_MODEL_DICT_TYPE,
    )
    from app.service.dict_service import DICT_VALUE_CACHE_PREFIX

    item = await dict_repository.get_by_type_code_and_name(
        db, DEFAULT_MODEL_DICT_TYPE, DEFAULT_MODEL_DICT_NAME
    )
    if item is None:
        if value is not None:
            db.add(
                SysDict(
                    type_code=DEFAULT_MODEL_DICT_TYPE,
                    name=DEFAULT_MODEL_DICT_NAME,
                    value=value,
                    status=status,
                    sort=1,
                    defaulted=1,
                    remark="平台默认对话模型（测试构造）",
                )
            )
    elif value is None:
        item.deleted = item.id
    else:
        item.value = value
        item.status = status
    await db.flush()
    await CacheService(redis).delete(
        f"{DICT_VALUE_CACHE_PREFIX}{DEFAULT_MODEL_DICT_TYPE}:{DEFAULT_MODEL_DICT_NAME}"
    )
