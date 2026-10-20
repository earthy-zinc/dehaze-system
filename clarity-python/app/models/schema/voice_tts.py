"""
语音交互模块 - TTS Schema 模型

音频格式与采样率白名单，以及**按供应商协议族**返回的可用音色目录。
内置引擎（openai_audio，clarity-voice）音色对应本地 Piper 模型；
云厂商音色为 vendor 侧定义（voice 即厂商 voice_type 透传），此处按协议族给出前端可选清单。
"""

from pydantic import BaseModel, Field

# 支持的音频输出格式（本地 Piper 引擎支持 mp3/wav/pcm）
FORMAT_VALUES = ("mp3", "wav", "pcm")
# 支持的采样率（Hz）
SAMPLE_RATE_VALUES = (8000, 16000, 24000, 48000)

# 内置引擎（Piper zh_CN-huayan-medium，默认音色 huayan）音色目录
_BUILTIN_VOICE_CATALOG = [
    {
        "id": "huayan",
        "name": "华燕",
        "description": "中文女声，清晰自然",
        "tags": ["女声", "中文"],
    },
]

# 火山引擎（volc_realtime）音色目录：id 即火山 voice_type，取「豆包语音合成模型 2.0」官方音色
# （uranus_bigtts 系列）。厂商音色清单随控制台版本演进，新增音色在此补充即可。
_VOLC_VOICE_CATALOG = [
    {
        "id": "zh_female_vv_uranus_bigtts",
        "name": "Vivi 2.0",
        "description": "多语种女声，支持方言",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_female_xiaohe_uranus_bigtts",
        "name": "小何 2.0",
        "description": "中文女声",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_male_m191_uranus_bigtts",
        "name": "云舟 2.0",
        "description": "中文男声",
        "tags": ["男声", "通用"],
    },
    {
        "id": "zh_male_taocheng_uranus_bigtts",
        "name": "小天 2.0",
        "description": "中文男声",
        "tags": ["男声", "通用"],
    },
    {
        "id": "zh_female_qingxinnvsheng_uranus_bigtts",
        "name": "清新女声 2.0",
        "description": "中文女声",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_female_cancan_uranus_bigtts",
        "name": "知性灿灿 2.0",
        "description": "中文女声，角色扮演",
        "tags": ["女声", "角色扮演"],
    },
    {
        "id": "zh_female_tianmeitaozi_uranus_bigtts",
        "name": "甜美桃子 2.0",
        "description": "中文女声",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_female_shuangkuaisisi_uranus_bigtts",
        "name": "爽快思思 2.0",
        "description": "中文女声",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_female_linjianvhai_uranus_bigtts",
        "name": "邻家女孩 2.0",
        "description": "中文女声",
        "tags": ["女声", "通用"],
    },
    {
        "id": "zh_male_sunwukong_uranus_bigtts",
        "name": "猴哥 2.0",
        "description": "中文男声，视频配音",
        "tags": ["男声", "视频配音"],
    },
]

# 协议族 → 音色目录；未登记（含内置 openai_audio 与未知协议）回落到内置引擎目录
_VOICE_CATALOG_BY_PROTOCOL: dict[str, list[dict[str, str]]] = {
    "volc_realtime": _VOLC_VOICE_CATALOG,
}


def get_voice_catalog(protocol_type: str | None) -> list[dict[str, str]]:
    """按默认 tts 模型所属供应商的协议族返回可用音色目录（未知协议回落内置引擎目录）"""
    return _VOICE_CATALOG_BY_PROTOCOL.get(protocol_type or "", _BUILTIN_VOICE_CATALOG)


def is_supported_voice(protocol_type: str | None, voice: str) -> bool:
    """voice 是否属于该协议族的音色目录"""
    return voice in {v["id"] for v in get_voice_catalog(protocol_type)}


class TtsForm(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="待合成文本(最长10000字符)")
    voice: str | None = Field(
        default=None,
        max_length=128,
        description="音色(缺省用平台默认音色，须属于默认 tts 模型的音色目录)",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速(0.8慢/1.0正常/1.2快)")
    format: str = Field(default="mp3", description="音频格式(mp3/wav/pcm)")
    sampleRate: int = Field(default=16000, description="采样率(Hz)")
