"""
语音交互模块 - TTS Schema 模型

音频格式与采样率白名单、可用音色目录（供路由直接返回）。
音色对应本地 Piper 引擎模型（piper-voices 开源库）。
"""

from pydantic import BaseModel, Field

# 支持的音频输出格式（本地 Piper 引擎支持 mp3/wav/pcm）
FORMAT_VALUES = ("mp3", "wav", "pcm")
# 支持的采样率（Hz）
SAMPLE_RATE_VALUES = (8000, 16000, 24000, 48000)

# 可用音色目录（Piper zh_CN-huayan-medium，默认音色 huayan）
VOICE_CATALOG = [
    {
        "id": "huayan",
        "name": "华燕",
        "description": "中文女声，清晰自然",
        "tags": ["女声", "中文"],
    },
]


class TtsForm(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="待合成文本(最长10000字符)"
    )
    voice: str | None = Field(default=None, max_length=32, description="音色(默认huayan)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速(0.8慢/1.0正常/1.2快)")
    format: str = Field(default="mp3", description="音频格式(mp3/wav/pcm)")
    sampleRate: int = Field(default=16000, description="采样率(Hz)")
