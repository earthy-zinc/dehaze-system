import pytest

from app.core.exceptions import BusinessException
from app.service.prediction import cache as cache_mod
from app.service.prediction.cache import invalidate_prediction_cache


async def test_invalidate_deletes_matching_keys(mock_redis):
    """正常路径：删除匹配算法的缓存键，返回删除数量。"""
    await mock_redis.set("prediction:1:img1", "{}")
    await mock_redis.set("prediction:1:img2", "{}")
    await mock_redis.set("prediction:2:img3", "{}")

    deleted = await invalidate_prediction_cache(1)

    assert deleted == 2
    assert await mock_redis.exists("prediction:2:img3")


async def test_invalidate_no_keys_returns_zero(mock_redis):
    """无 key 可删属正常态，返回 0 而非失败。"""
    assert await invalidate_prediction_cache(99) == 0


async def test_invalidate_redis_failure_raises(monkeypatch):
    """Redis 删除失败不得静默当作已清理，须显式抛异常暴露。"""

    async def _force_fallback(operation, fallback=None, **kwargs):
        # 模拟 Redis 不可用时 wrapper 进入降级路径，直接执行 fallback
        return await fallback()

    monkeypatch.setattr(cache_mod, "redis_operation_with_fallback", _force_fallback)

    with pytest.raises(BusinessException) as ei:
        await invalidate_prediction_cache(1)
    assert "预测缓存失效失败" in ei.value.message
