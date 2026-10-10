"""权重解析：本地缓存优先（降级口径）、下载、失败语义。"""

from pathlib import Path

import httpx
import pytest
import respx

from algorithm import model_loader
from app.config import settings

WEIGHT_PATH = "AECR-Net/NH_train.pk"
WEIGHT_URL = "http://127.0.0.1:9000/models/AECR-Net/NH_train.pk"


@respx.mock
def test_cached_weight_is_used_without_download():
    """缓存命中即降级使用本地文件：不进网络"""
    cache_path = Path(settings.MODEL_CACHE_DIR) / WEIGHT_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"CACHED")
    route = respx.get(WEIGHT_URL).mock(return_value=httpx.Response(200, content=b"REMOTE"))

    assert model_loader.resolve_model_path(WEIGHT_PATH) == str(cache_path)
    assert not route.called


@respx.mock
def test_missing_weight_is_downloaded_to_cache():
    route = respx.get(WEIGHT_URL).mock(return_value=httpx.Response(200, content=b"WEIGHTS"))

    local_path = model_loader.resolve_model_path(WEIGHT_PATH)

    assert route.called
    assert Path(local_path).read_bytes() == b"WEIGHTS"


@respx.mock
def test_download_failure_raises_and_leaves_no_partial_file():
    respx.get(WEIGHT_URL).mock(return_value=httpx.Response(500))

    with pytest.raises(FileNotFoundError):
        model_loader.resolve_model_path(WEIGHT_PATH)

    cache_dir = Path(settings.MODEL_CACHE_DIR) / "AECR-Net"
    assert not (cache_dir / "NH_train.pk").exists()
    assert not (cache_dir / "NH_train.pk.part").exists()


def test_empty_weight_path_raises():
    with pytest.raises(FileNotFoundError):
        model_loader.resolve_model_path("  ")


def test_check_model_exists_reports_cached_size():
    cache_path = Path(settings.MODEL_CACHE_DIR) / WEIGHT_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"x" * 42)

    assert model_loader.check_model_exists(WEIGHT_PATH) == 42


@respx.mock
def test_check_model_exists_falls_back_to_head_content_length():
    respx.head(WEIGHT_URL).mock(return_value=httpx.Response(200, headers={"Content-Length": "99"}))

    assert model_loader.check_model_exists(WEIGHT_PATH) == 99


@respx.mock
def test_check_model_exists_returns_none_when_unreachable():
    respx.head(WEIGHT_URL).mock(return_value=httpx.Response(404))

    assert model_loader.check_model_exists(WEIGHT_PATH) is None
