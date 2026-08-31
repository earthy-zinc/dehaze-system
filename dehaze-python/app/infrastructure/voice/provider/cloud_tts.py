"""云端 TTS Provider：基于 CloudBase 通用框架，厂商协议适配待接入

通用能力（API Key 选取、认证头构建、HTTP 请求）由 CloudBase 提供。
具体厂商协议（合成端点/请求体/音色映射、返回音频格式）差异大且需厂商
API 规格，在此以清晰方法边界保留扩展点：拿到厂商 API 规格后逐厂商实现。
"""

from app.infrastructure.voice.provider.base import TTSProvider
from app.infrastructure.voice.provider.cloud_base import CloudBase


class CloudTtsProvider(TTSProvider, CloudBase):
    """云端语音合成 Provider（Azure/阿里云）——通用框架就绪，协议适配待接入"""

    def __init__(self, provider) -> None:
        CloudBase.__init__(self, provider)

    async def synthesize(
        self, text: str, voice_id: str | None, speed: float, format_: str, sample_rate: int
    ) -> bytes:
        raise NotImplementedError("云端 TTS 合成：需厂商 API 规格（合成端点/音色映射/音频格式）适配")

    async def engine_status(self) -> dict:
        return {
            "engine_status": "offline",
            "engine": self._provider.provider_code,
            "remark": "云端 TTS 厂商协议适配待实现（需厂商 API 规格）",
        }
