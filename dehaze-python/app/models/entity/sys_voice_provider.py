from typing import Any

from sqlalchemy import BigInteger, Index, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysVoiceProvider(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_voice_provider"
    __table_args__ = (
        Index("idx_engine_default", "engine_type", "is_default"),
        Index("idx_engine_status", "engine_type", "status"),
        {"comment": "语音引擎供应商配置表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_code: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
        comment="引擎编码(local;aliyun;tencent;xfyun;azure;删除后不可复用)",
    )
    engine_type: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="能力类型(asr:语音识别;tts:语音合成)"
    )
    display_name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="显示名称(如本地FunASR;阿里云ASR;本地Piper;Azure TTS)"
    )
    api_base_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="引擎API基础地址(local为空,走进程内引擎)"
    )
    auth_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="bearer",
        comment="认证方式(bearer:Authorization Bearer;x-api-key;custom:自定义请求头,头名在default_headers配置)",
    )
    default_headers: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment='默认请求头(JSON);auth_type=custom时,需含{"auth_header":"头名"}'
    )
    is_default: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="该engine_type维度下默认引擎(0:否;1:是;每能力维度仅一条为1)",
    )
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="排序序号(数字越小越靠前)"
    )
    health_check_enabled: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="健康检查开关(1:开启,参与熔断判定;0:关闭)",
    )
    remark: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="运维备注(账号归属/合同号/商务信息)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
