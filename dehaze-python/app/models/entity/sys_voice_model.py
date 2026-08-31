from typing import Any

from sqlalchemy import BigInteger, Index, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysVoiceModel(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_voice_model"
    __table_args__ = (
        Index("idx_engine_type", "engine_type", "model_type", "status"),
        {"comment": "语音引擎模型/音色注册表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联引擎ID(关联sys_voice_provider.id)"
    )
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="模型/音色业务编码(sensevoice;paraformer;huayan;删除后不可复用)"
    )
    engine_type: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="能力类型(asr:语音识别;tts:语音合成)"
    )
    model_type: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="子类型(ASR:stream流式/offline离线;TTS:voice音色)"
    )
    display_name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="显示名称(如中文女声;SenseVoice流式)"
    )
    params: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="模型参数(JSON:本地模型路径/下载URL/推理参数;云端厂商模型ID/采样率/编码等)",
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
