"""模型下载与校验（mock 网络）：镜像回退、断点续传、字节校验、原子生效。"""

import httpx
import pytest
import respx

from app import model_store
from app.config import settings

URL_MIRROR = "https://hf-mirror.com/x/model.gguf"
URL_OFFICIAL = "https://huggingface.co/x/model.gguf"
CONTENT = b"12345678"


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    """把对话模型指向临时目录，并把预期大小缩小到 8 字节以便测试。"""
    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "MODEL_PATH", "")
    monkeypatch.setattr(model_store, "CHAT_MODEL_SIZE", len(CONTENT))
    monkeypatch.setattr(model_store, "_CHAT_URLS", [URL_MIRROR, URL_OFFICIAL])
    return tmp_path


def test_ensure_short_circuits_when_file_ready(cache_dir):
    path = cache_dir / model_store.CHAT_MODEL_FILE
    path.write_bytes(CONTENT)
    with respx.mock:
        # 未注册任何路由：一旦发起下载即失败
        assert model_store.ensure_chat_model() == str(path)


def test_download_finalizes_atomically(cache_dir):
    with respx.mock:
        respx.get(URL_MIRROR).mock(return_value=httpx.Response(200, content=CONTENT))
        result = model_store.ensure_chat_model()

    assert result == str(cache_dir / model_store.CHAT_MODEL_FILE)
    assert (cache_dir / model_store.CHAT_MODEL_FILE).read_bytes() == CONTENT
    assert not (cache_dir / (model_store.CHAT_MODEL_FILE + ".part")).exists()
    assert model_store.is_chat_downloaded() is True


def test_falls_back_to_second_mirror(cache_dir):
    with respx.mock:
        respx.get(URL_MIRROR).mock(return_value=httpx.Response(503))
        respx.get(URL_OFFICIAL).mock(return_value=httpx.Response(200, content=CONTENT))
        model_store.ensure_chat_model()

    assert (cache_dir / model_store.CHAT_MODEL_FILE).read_bytes() == CONTENT


def test_resumes_from_part_file(cache_dir):
    part = cache_dir / (model_store.CHAT_MODEL_FILE + ".part")
    part.write_bytes(CONTENT[:4])
    with respx.mock:
        route = respx.get(URL_MIRROR).mock(return_value=httpx.Response(206, content=CONTENT[4:]))
        model_store.ensure_chat_model()

    assert route.calls[0].request.headers["range"] == "bytes=4-"
    assert (cache_dir / model_store.CHAT_MODEL_FILE).read_bytes() == CONTENT


def test_remote_size_mismatch_raises(cache_dir):
    with respx.mock:
        respx.get(URL_MIRROR).mock(return_value=httpx.Response(200, content=b"12345"))
        respx.get(URL_OFFICIAL).mock(return_value=httpx.Response(200, content=b"12345"))
        with pytest.raises(RuntimeError, match="模型自动下载失败"):
            model_store.ensure_chat_model()

    assert not (cache_dir / model_store.CHAT_MODEL_FILE).exists()
    assert not (cache_dir / (model_store.CHAT_MODEL_FILE + ".part")).exists()


def test_incomplete_download_raises(cache_dir):
    with respx.mock:
        respx.get(URL_MIRROR).mock(
            return_value=httpx.Response(200, content=CONTENT, headers={"content-length": "16"})
        )
        respx.get(URL_OFFICIAL).mock(
            return_value=httpx.Response(200, content=CONTENT, headers={"content-length": "16"})
        )
        with pytest.raises(RuntimeError, match="模型自动下载失败"):
            model_store.ensure_chat_model()


def test_rerank_readiness_requires_exact_size(cache_dir):
    path = cache_dir / model_store.RERANK_MODEL_FILE
    path.write_bytes(b"x" * (model_store.RERANK_MODEL_SIZE - 1))
    assert model_store.is_rerank_downloaded() is False
    with path.open("wb") as f:
        f.truncate(model_store.RERANK_MODEL_SIZE)
    assert model_store.is_rerank_downloaded() is True
    assert model_store.is_embedding_downloaded() is False
