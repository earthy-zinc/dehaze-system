"""引擎错误对外文案 + 语音状态探测超时。

引擎错误码/异常原文可能携带内网地址与 httpx 异常尾缀：对外文案截断到业务首段，
完整原文只进日志；语音状态探测独立短超时，避免引擎挂起拖垮监控接口。
"""

import json
import logging

import pytest

from app.infrastructure.llm.client import openai_audio_client as audio_client
from app.infrastructure.llm.client import openai_image_client as image_client

_WEIGHT_ERROR = (
    "模型权重不可用: 模型权重文件不可用: path=AECR-Net/NH_train.pk, "
    "url=http://earthyzinc.devcloud.woa.com:9000/models/AECR-Net/NH_train.pk, "
    "error=Client error '404 Not Found' for url 'http://earthyzinc.devcloud.woa.com:9000/x'\n"
    "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
)


def _body(message: str) -> bytes:
    return json.dumps({"code": "MODEL_UNAVAILABLE", "message": message}).encode()


def test_engine_error_strips_internal_url_and_httpx_tail(caplog):
    with caplog.at_level(logging.ERROR), pytest.raises(image_client.BusinessException) as exc:
        image_client._raise_engine_error(500, _body(_WEIGHT_ERROR))

    assert (
        exc.value.message
        == "图像引擎服务错误：模型权重不可用: 模型权重文件不可用: path=AECR-Net/NH_train.pk"
    )
    assert "earthyzinc.devcloud.woa.com" not in exc.value.message
    assert "developer.mozilla.org" not in exc.value.message
    # 完整原文只进日志，不进响应体
    assert "earthyzinc.devcloud.woa.com" in caplog.text
    assert "developer.mozilla.org" in caplog.text


def test_engine_error_keeps_plain_message_untouched():
    with pytest.raises(image_client.BusinessException) as exc:
        image_client._raise_engine_error(500, _body("内部错误"))

    assert exc.value.message == "图像引擎服务错误：内部错误"


def test_engine_error_cuts_at_error_segment_without_url():
    with pytest.raises(image_client.BusinessException) as exc:
        image_client._raise_engine_error(500, _body("推理失败, error=RuntimeError: cuda oom"))

    assert exc.value.message == "图像引擎服务错误：推理失败"


def test_voice_engine_error_strips_internal_url_and_httpx_tail(caplog):
    """语音侧与图像侧同规则：对外截断业务首段，完整原文只进日志"""
    with caplog.at_level(logging.ERROR), pytest.raises(audio_client.BusinessException) as exc:
        audio_client._raise_engine_error(500, _body(_WEIGHT_ERROR))

    assert (
        exc.value.message
        == "语音引擎服务错误：模型权重不可用: 模型权重文件不可用: path=AECR-Net/NH_train.pk"
    )
    assert "earthyzinc.devcloud.woa.com" not in exc.value.message
    assert "earthyzinc.devcloud.woa.com" in caplog.text


async def test_voice_status_probe_uses_short_timeout(monkeypatch):
    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"asr": {"loaded": True}, "tts": {"loaded": False}}

    async def _fake_request(method, path, provider, timeout=audio_client._HTTP_TIMEOUT, **kwargs):
        captured["path"] = path
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(audio_client, "_request", _fake_request)

    result = await audio_client.status(object())

    assert captured["path"] == "/status"
    assert captured["timeout"] == audio_client._READY_TIMEOUT
    assert result == {
        "engine_status": "online",
        "stream_model_loaded": True,
        "offline_model_loaded": True,
        "voice_model_loaded": False,
    }


async def test_voice_status_tolerates_missing_endpoint(monkeypatch):
    """供应商无 /status 端点（404）：按在线 + 加载态未知容错上报，不因"无探针"误报 offline。"""
    captured: dict = {}

    class _Resp:
        status_code = 404

    async def _fake_request(
        method, path, provider, timeout=audio_client._HTTP_TIMEOUT, tolerate_not_found=False, **kw
    ):
        captured["tolerate_not_found"] = tolerate_not_found
        return _Resp()

    monkeypatch.setattr(audio_client, "_request", _fake_request)

    result = await audio_client.status(object())

    assert captured["tolerate_not_found"] is True  # 状态探测显式开启 404 容错
    assert result == {
        "engine_status": "online",
        "stream_model_loaded": False,
        "offline_model_loaded": False,
        "voice_model_loaded": False,
    }
