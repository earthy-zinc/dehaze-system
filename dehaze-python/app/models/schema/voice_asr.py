"""
语音交互模块 - ASR Schema 模型
"""

from pydantic import BaseModel, Field


class StreamAsrSessionForm(BaseModel):
    model: str | None = Field(
        default=None, max_length=64, description="ASR模型(默认流式模型sensevoice)"
    )
