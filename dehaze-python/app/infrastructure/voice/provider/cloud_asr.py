"""云端 ASR Provider：基于 CloudBase 通用框架，厂商协议适配待接入

通用能力（API Key 选取、认证头构建、HTTP 请求）由 CloudBase 提供。
具体厂商协议（流式 WebSocket、离线端点/请求体、热词 API）差异大且需厂商
API 规格，在此以清晰方法边界保留扩展点：拿到厂商 API 规格后逐厂商实现。
"""

from typing import AsyncIterator

from app.infrastructure.voice.provider.base import ASRProvider, ASRStreamSession
from app.infrastructure.voice.provider.cloud_base import CloudBase


class CloudAsrProvider(ASRProvider, CloudBase):
    """云端语音识别 Provider（阿里云/腾讯云/讯飞）——通用框架就绪，协议适配待接入"""

    def __init__(self, provider) -> None:
        CloudBase.__init__(self, provider)

    async def recognize_stream(self) -> ASRStreamSession:
        raise NotImplementedError("云端 ASR 流式识别：需厂商 API 规格（WebSocket 流式协议）适配")

    async def recognize_offline(self, audio_bytes: bytes) -> str:
        raise NotImplementedError("云端 ASR 离线识别：需厂商 API 规格（离线端点/请求体/响应解析）适配")

    async def register_hotwords(self, words: list[str]) -> None:
        raise NotImplementedError("云端 ASR 热词注册：需厂商 API 规格适配")

    async def engine_status(self) -> dict:
        return {
            "engine_status": "offline",
            "engine": self._provider.provider_code,
            "remark": "云端 ASR 厂商协议适配待实现（需厂商 API 规格）",
        }
