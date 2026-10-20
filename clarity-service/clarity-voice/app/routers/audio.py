"""OpenAI Audio 兼容端点：`/v1/audio/speech`、`/v1/audio/transcriptions`。

字段名对齐 OpenAI：`model`/`input`/`voice`/`response_format`/`speed`，multipart 用
`file`/`model`；本服务扩展字段放**顶层**（`sample_rate`、`voice_params`）。
`model` 的取值是统一模型表的 `model_id`（`huayan`/`sensevoice`/`paraformer`），
真实推理工件由服务端目录（`app.catalog`）解析，编排侧不必再下发；
`response_format` 仅支持 `mp3/wav/pcm`（本服务是 OpenAI Audio 的子集）。
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, File, Form, Response, UploadFile
from pydantic import BaseModel, field_validator

from app import catalog
from app.config import settings
from app.engines import asr as asr_engine
from app.engines import tts as tts_engine
from app.engines.tts import SUPPORTED_FORMATS, SUPPORTED_SAMPLE_RATES, VoiceParams
from app.errors import ParamError
from app.metrics import INFERENCE_DURATION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["audio"])

_MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/L16"}
_DEFAULT_SAMPLE_RATE = 22050


class SpeechRequest(BaseModel):
    """`POST /v1/audio/speech` 请求体（OpenAI 字段 + 顶层扩展）"""

    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    # 可选调试覆盖：缺省时由服务端目录按 (model, voice) 解析
    voice_params: VoiceParams | None = None

    @field_validator("input")
    @classmethod
    def _input_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input 不能为空")
        if len(value) > settings.TTS_MAX_TEXT_CHARS:
            raise ValueError(f"input 超过上限 {settings.TTS_MAX_TEXT_CHARS} 字符")
        return value

    @field_validator("speed")
    @classmethod
    def _speed_range(cls, value: float) -> float:
        if not 0.5 <= value <= 2.0:
            raise ValueError("speed 必须在 0.5~2.0 之间")
        return value

    @field_validator("response_format")
    @classmethod
    def _format_supported(cls, value: str) -> str:
        if value not in SUPPORTED_FORMATS:
            raise ValueError(f"response_format 仅支持 {'/'.join(SUPPORTED_FORMATS)}")
        return value

    @field_validator("sample_rate")
    @classmethod
    def _sample_rate_supported(cls, value: int) -> int:
        if value not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate 仅支持 {'/'.join(str(rate) for rate in SUPPORTED_SAMPLE_RATES)}"
            )
        return value


@router.post("/speech")
async def speech(payload: SpeechRequest) -> Response:
    """合成语音：按 model/voice 解析权重来源（可用 voice_params 覆盖），返回音频字节"""
    params = payload.voice_params or catalog.resolve_voice(payload.model, payload.voice)
    started = time.perf_counter()
    audio = await tts_engine.run_synthesize(
        payload.input,
        payload.voice,
        params,
        payload.speed,
        payload.response_format,
        payload.sample_rate,
    )
    elapsed = time.perf_counter() - started
    INFERENCE_DURATION.labels("tts").observe(elapsed)
    logger.info(
        "语音合成完成",
        extra={
            "model": payload.model,
            "voice": payload.voice,
            "response_format": payload.response_format,
            "sample_rate": payload.sample_rate,
            "chars": len(payload.input),
            "audio_bytes": len(audio),
            "duration_ms": round(elapsed * 1000, 3),
        },
    )
    return Response(content=audio, media_type=_MEDIA_TYPES[payload.response_format])


@router.post("/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    hotwords: str | None = Form(None),
) -> dict[str, str]:
    """离线识别整段音频（WAV/PCM，16kHz/16bit/mono）→ {text}"""
    data = await file.read()
    if not data:
        raise ParamError("音频文件为空")
    model_id = catalog.resolve_asr_model(model)
    started = time.perf_counter()
    text = await asr_engine.run_transcribe(data, model_id, _parse_hotwords(hotwords))
    elapsed = time.perf_counter() - started
    INFERENCE_DURATION.labels("asr").observe(elapsed)
    logger.info(
        "离线识别完成",
        extra={
            "model": model,
            "model_id": model_id,
            "audio_bytes": len(data),
            "duration_ms": round(elapsed * 1000, 3),
        },
    )
    return {"text": text}


def _parse_hotwords(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    try:
        words = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParamError(f"hotwords 不是合法 JSON 数组: {e}") from e
    if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
        raise ParamError("hotwords 必须为字符串数组")
    return [word for word in words if word.strip()]
