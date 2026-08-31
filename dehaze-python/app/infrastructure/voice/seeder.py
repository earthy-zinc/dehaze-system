"""内置本地语音引擎的数据播种（幂等）

启动时确保 local asr/tts provider 与默认模型/音色注册到语音引擎注册表，
使本地 FunASR（ASR）与 Piper（TTS）开箱即用：

- local asr provider：默认 ASR 引擎，本地 FunASR；
- local tts provider：默认 TTS 引擎，本地 Piper；
- sys_voice_model：为 local 引擎注册 SenseVoice 流式 / Paraformer 离线 ASR
  模型与 huayan 中文女声 TTS 音色，params 携带本地模型/音色信息，
  供 funasr_engine/piper_tts_engine 从注册表解析。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity import SysVoiceModel, SysVoiceProvider
from app.repository.voice_model_repository import voice_model_repository
from app.repository.voice_provider_repository import voice_provider_repository

logger = logging.getLogger(__name__)

_LOCAL_PROVIDER_CODE = "local"

# 各能力维度的 local 引擎（provider_code 在 engine_type 维度内唯一）
_LOCAL_PROVIDERS = [
    {"engine_type": "asr", "display_name": "本地FunASR"},
    {"engine_type": "tts", "display_name": "本地Piper"},
]

# 默认播种的本地 ASR 模型 / TTS 音色（params 为模型参数 JSON）
_LOCAL_MODELS = [
    {
        "model_id": "sensevoice",
        "engine_type": "asr",
        "model_type": "stream",
        "display_name": "SenseVoice流式",
        "params": {"model_id": "iic/SenseVoiceSmall"},
    },
    {
        "model_id": "paraformer",
        "engine_type": "asr",
        "model_type": "offline",
        "display_name": "Paraformer离线",
        "params": {
            "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        },
    },
    {
        "model_id": "huayan",
        "engine_type": "tts",
        "model_type": "voice",
        "display_name": "中文女声",
        "params": {
            "onnx": "zh_CN-huayan-medium.onnx",
            "size": 63201294,
            "urls": [
                "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{filename}",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{filename}",
            ],
        },
    },
]


async def ensure_local_engines(db: AsyncSession) -> None:
    """幂等播种 local asr/tts provider 与默认模型/音色；已存在则仅补齐 is_default/status"""
    for spec in _LOCAL_PROVIDERS:
        engine_type = spec["engine_type"]
        provider = await voice_provider_repository.get_by_provider_and_engine(
            db, _LOCAL_PROVIDER_CODE, engine_type
        )
        if provider is None:
            provider = SysVoiceProvider(
                provider_code=_LOCAL_PROVIDER_CODE,
                engine_type=engine_type,
                display_name=spec["display_name"],
                api_base_url=None,
                auth_type="bearer",
                is_default=1,
                status=1,
            )
            db.add(provider)
            await db.flush()
            logger.info("播种 local %s provider（id=%s）", engine_type, provider.id)
        else:
            # 已存在则补齐默认与启用状态，确保本地引擎始终为默认可用
            if provider.is_default != 1 or provider.status != 1:
                provider.is_default = 1
                provider.status = 1
                await db.flush()

        for model in _LOCAL_MODELS:
            if model["engine_type"] != engine_type:
                continue
            existing = await voice_model_repository.get_by_model_and_provider(
                db, model["model_id"], provider.id
            )
            if existing is None:
                db.add(
                    SysVoiceModel(
                        provider_id=provider.id,
                        model_id=model["model_id"],
                        engine_type=model["engine_type"],
                        model_type=model["model_type"],
                        display_name=model["display_name"],
                        params=model["params"],
                        status=1,
                    )
                )
                await db.flush()
                logger.info("播种 local %s 模型/音色 %s", engine_type, model["model_id"])
            elif (existing.params or {}) != model["params"]:
                # 内置种子参数演进（如音色下载地址）时幂等补齐
                existing.params = model["params"]
                await db.flush()
                logger.info("更新 local %s 模型/音色 %s 参数", engine_type, model["model_id"])
