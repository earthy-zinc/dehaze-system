"""图像结果缓存测试（统一模型口径）。

- 缓存键为 `image:{model_id}:{imageMd5}`（与模型绑定，换模型不串用）
- 缓存只存与环境无关的对象标识（storage + objectName）+ resultMd5，
  命中时按当前配置 baseUrl 运行时拼接 URL——存储域名随环境变化后，
  缓存命中不能再回放历史环境写死的旧地址
"""

import hashlib
import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest

from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction import prediction_service as prediction_module
from app.service.prediction.cache import get_cached_prediction, set_cached_prediction
from app.service.prediction.image_route import resolve_image_model
from app.service.prediction.prediction_service import prediction_service

MODEL_ID = "aecrnet"
_IMAGE_BYTES = b"fake-png-payload"
_IMAGE_MD5 = hashlib.md5(_IMAGE_BYTES).hexdigest()
_RESULT_MD5 = "c" * 32
_CACHE_KEY = f"image:{MODEL_ID}:{_IMAGE_MD5}"


def _patch_image(monkeypatch) -> None:
    async def _fetch(url):
        return BytesIO(_IMAGE_BYTES)

    monkeypatch.setattr(prediction_module, "fetch_image", _fetch)


# ===== 读写语义 =====


async def test_set_and_get_cached_prediction_roundtrip(mock_redis):
    await set_cached_prediction(_CACHE_KEY, {"storage": "local", "resultMd5": _RESULT_MD5})
    assert await get_cached_prediction(_CACHE_KEY) == {
        "storage": "local",
        "resultMd5": _RESULT_MD5,
    }
    assert await get_cached_prediction(f"image:other:{_IMAGE_MD5}") is None


# ===== 缓存命中：URL 按当前配置运行时拼接 =====


@pytest.mark.requires_db
async def test_cache_hit_rebuilds_url_from_current_storage_config(db, mock_redis, monkeypatch):
    """命中缓存时按当前配置 baseUrl + objectName 重拼 URL，忽略缓存里的历史域名。"""
    _patch_image(monkeypatch)
    object_name = "predictions/20200101/x.png"
    stale_url = f"http://stale-internal.example:9000/{object_name}"
    await mock_redis.set(
        _CACHE_KEY,
        json.dumps(
            {
                "storage": "local",
                "objectName": object_name,
                "resultUrl": stale_url,
                "resultMd5": _RESULT_MD5,
            }
        ),
    )

    result = await prediction_service.predict(
        model=MODEL_ID,
        image_url="https://cdn.example.com/a.png",
    )

    assert result["status"] == LogStatus.COMPLETED.value
    assert result["resultMd5"] == _RESULT_MD5
    assert object_name in result["resultUrl"]
    assert "stale-internal.example" not in result["resultUrl"]

    log = await pred_log_repository.get_by_id(db, result["taskId"])
    assert log is not None
    assert log.model_id == MODEL_ID
    assert log.pred_md5 == _RESULT_MD5


@pytest.mark.requires_db
async def test_cache_value_without_object_name_falls_back_to_recompute(db, mock_redis, monkeypatch):
    """旧格式缓存（只有 resultUrl、无 objectName）视同未命中，走重算而非回放旧 URL。"""
    _patch_image(monkeypatch)
    monkeypatch.setattr(prediction_service, "_execute_async", AsyncMock())
    await mock_redis.set(
        _CACHE_KEY,
        json.dumps({"resultUrl": "http://stale-internal.example/x.png", "resultMd5": _RESULT_MD5}),
    )

    result = await prediction_service.predict(
        model=MODEL_ID,
        image_url="https://cdn.example.com/a.png",
    )

    assert result["status"] == LogStatus.PROCESSING.value
    assert "resultUrl" not in result
    log = await pred_log_repository.get_by_id(db, result["taskId"])
    assert log is not None
    assert log.status == LogStatus.PROCESSING.value


# ===== 异步完成：写缓存只存环境无关标识 =====


@pytest.mark.requires_db
async def test_async_result_cache_stores_no_absolute_url(db, mock_redis, monkeypatch):
    """异步推理完成后写缓存只存 storage + objectName + resultMd5，不写绝对 URL。"""
    log = await pred_log_repository.create_pending_log(
        db,
        model_id=MODEL_ID,
        origin_md5=_IMAGE_MD5,
        origin_url="https://cdn.example.com/a.png",
    )
    result_object_name = "predictions/20260101/out.png"
    route = await resolve_image_model(db, MODEL_ID)

    async def _fake_run_edit(r, image_bytes, params):
        return _IMAGE_BYTES

    monkeypatch.setattr(prediction_module, "run_edit", _fake_run_edit)
    monkeypatch.setattr(
        prediction_module, "upload_result", AsyncMock(return_value=result_object_name)
    )

    await prediction_service._execute_async(
        log_id=log.id,
        route=route,
        image_bytes=BytesIO(_IMAGE_BYTES),
        image_md5=_IMAGE_MD5,
        cache_key=_CACHE_KEY,
    )

    raw = await mock_redis.get(_CACHE_KEY)
    assert raw is not None
    cached = json.loads(raw)
    assert "resultUrl" not in cached
    assert cached["objectName"] == result_object_name
    assert cached["resultMd5"] == hashlib.md5(_IMAGE_BYTES).hexdigest()
