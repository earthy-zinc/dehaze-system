"""数据工厂：纯数据构造函数（redis/orm/conv/member/benefit/context 等）。

本模块只放"构造测试数据"的纯函数（factories），不含协议仿真或边界 patch。
协议仿真见 tests.stubs.fakes，patch 辅助见 tests.stubs.mocks。
"""

from datetime import datetime
from types import SimpleNamespace

from fakeredis import FakeAsyncRedis


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
