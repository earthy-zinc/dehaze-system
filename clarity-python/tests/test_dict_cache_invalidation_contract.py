"""字典缓存失效口径的跨端契约（免库，fakeredis）。

背景：三端共享同一 Redis，但读缓存用的键不同——java/go 读 `dict:data:{typeCode}`
（下拉选项），python 业务读路径读 `dict:value:{typeCode}:{key}`
（`dict_service.get_dict_value`，平台默认模型解析 / 会员规则 / 收藏容量等）。

若 java/go 改字典后只失效 `dict:data:`，python 侧判定将最长滞后 1 小时
（L2 TTL 3600s；且 L1 回填时继承 L2 剩余 TTL）。故**失效键集合必须统一覆盖两个前缀**。

本用例锚定 python 侧口径（写路径同一钩子失效两个前缀 + 按 typeCode 精确隔离），
并锁住管理端统一失效清单；go 侧同款对拍见
`clarity-go/internal/service/dict/sys_dict_cache_invalidation_test.go`。
"""

from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import run_after_commit_callbacks
from app.infrastructure.cache.cache import CacheService
from app.service.dict_service import (
    DICT_OPTIONS_CACHE_PREFIX,
    DICT_VALUE_CACHE_PREFIX,
    _defer_invalidate_dict_caches,
)

_TYPE_CODE = "ai_model"


class _FakeSession:
    """只提供 defer_after_commit 所需的 session.info 容器（免真实数据库会话）"""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {}


async def test_dict_write_invalidates_options_and_value_caches(mock_redis):
    """字典写路径提交后必须同时失效 `dict:data:{typeCode}` 与 `dict:value:{typeCode}:*`"""
    session = _FakeSession()
    options_key = f"{DICT_OPTIONS_CACHE_PREFIX}{_TYPE_CODE}"
    value_key = f"{DICT_VALUE_CACHE_PREFIX}{_TYPE_CODE}:default_model"
    neighbor_type_key = f"{DICT_VALUE_CACHE_PREFIX}member_growth_rules:sign_in_value"

    cache = CacheService(mock_redis)
    await cache.set_json(options_key, [{"value": "1", "label": "default_model"}], 3600)
    await cache.set_json(value_key, "qwen3-0.6b", 3600)
    await cache.set_json(neighbor_type_key, "3", 3600)

    _defer_invalidate_dict_caches(cast(AsyncSession, session), mock_redis, _TYPE_CODE)
    await run_after_commit_callbacks(cast(AsyncSession, session))

    # L2（共享 Redis）两侧都必须被删掉
    assert await mock_redis.get(options_key) is None
    assert await mock_redis.get(value_key) is None
    # 进程内 L1 同步清掉（否则本实例仍会命中旧值到 TTL 到期）
    assert await cache.get_json(options_key) is None
    assert await cache.get_json(value_key) is None
    # 失效范围按 typeCode 精确隔离：同前缀的其它类型字典值不得被误删
    assert await cache.get_json(neighbor_type_key) == "3"


async def test_dict_value_invalidation_pattern_covers_all_keys_of_type(mock_redis):
    """`dict:value:{typeCode}:*` 是通配失效：同类型下多个键（如 default_model.asr）都要清"""
    session = _FakeSession()
    keys = [
        f"{DICT_VALUE_CACHE_PREFIX}{_TYPE_CODE}:default_model",
        f"{DICT_VALUE_CACHE_PREFIX}{_TYPE_CODE}:default_model.asr",
        f"{DICT_VALUE_CACHE_PREFIX}{_TYPE_CODE}:default_model.tts",
    ]
    cache = CacheService(mock_redis)
    for key in keys:
        await cache.set_json(key, "x", 3600)

    _defer_invalidate_dict_caches(cast(AsyncSession, session), mock_redis, _TYPE_CODE)
    await run_after_commit_callbacks(cast(AsyncSession, session))

    for key in keys:
        assert await mock_redis.get(key) is None


def test_admin_cache_clear_patterns_cover_both_dict_prefixes():
    """管理端统一失效入口的业务缓存清单必须同时覆盖两个前缀（否则运维清缓存漏一半）"""
    from app.router.cache import _BUSINESS_CACHE_PATTERNS

    assert "dict:data:*" in _BUSINESS_CACHE_PATTERNS
    assert "dict:value:*" in _BUSINESS_CACHE_PATTERNS


def test_dict_cache_key_prefixes_are_cross_end_contract():
    """键形态即跨端契约（java/go 失效同一批键，改这里必须同步三端）"""
    assert DICT_OPTIONS_CACHE_PREFIX == "dict:data:"
    assert DICT_VALUE_CACHE_PREFIX == "dict:value:"
