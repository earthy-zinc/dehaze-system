"""本地语音模型/音色目录（服务端解析：只读声明，不查库、不做业务缓存）。

阶段 2 起运行时工件不再落库（原 `sys_voice_model` 表已删），模型标识统一为
`sys_ai_model.model_id`（`huayan`/`sensevoice`/`paraformer`），引擎按 model_id
在本目录解析出真实推理工件（Piper onnx 权重来源 / ModelScope 模型标识）。
"""

from __future__ import annotations

from app.engines.tts import VoiceParams
from app.errors import ParamError

# TTS 音色目录：model_id → {voice_id: 权重来源（onnx 文件名/size/下载镜像模板）}
VOICE_CATALOG: dict[str, dict[str, VoiceParams]] = {
    "huayan": {
        "huayan": VoiceParams(
            onnx="zh_CN-huayan-medium.onnx",
            size=63201294,
            urls=[
                (
                    "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/"
                    "zh/zh_CN/huayan/medium/{filename}"
                ),
                (
                    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "zh/zh_CN/huayan/medium/{filename}"
                ),
            ],
        )
    }
}

# ASR 模型目录：统一 model_id → ModelScope 模型标识
ASR_CATALOG: dict[str, str] = {
    "sensevoice": "iic/SenseVoiceSmall",
    "paraformer": "iic/speech_seaco_paraformer_large_asr_nat-zh",
}


def resolve_voice(model: str, voice: str) -> VoiceParams:
    """按 (model_id, voice_id) 解析音色权重来源；未登记即 400（不做静默回退）"""
    voices = VOICE_CATALOG.get(model)
    if voices is None:
        raise ParamError(f"不支持的 tts 模型: {model}")
    params = voices.get(voice)
    if params is None:
        raise ParamError(f"tts 模型 {model} 不支持的音色: {voice}")
    return params


def resolve_asr_model(model: str) -> str:
    """按统一 model_id 解析 ModelScope 模型标识；未登记即 400（不做静默回退）"""
    resolved = ASR_CATALOG.get(model)
    if resolved is None:
        raise ParamError(f"不支持的 asr 模型: {model}")
    return resolved


def catalog_snapshot() -> dict[str, object]:
    """能力发现用目录快照（/v1/models 暴露）"""
    return {
        "tts": {model: sorted(voices) for model, voices in VOICE_CATALOG.items()},
        "asr": sorted(ASR_CATALOG),
    }
