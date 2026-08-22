from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiProvider(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_provider"
    __table_args__ = {"comment": "AI模型供应商配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_code: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
        comment="供应商编码(openai;anthropic;deepseek;zhipu;qwen;custom)",
    )
    display_name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="显示名称(如OpenAI;Anthropic;DeepSeek)"
    )
    api_base_url: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="API基础地址(如https://api.openai.com/v1)"
    )
    protocol_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="openai_compat",
        comment="协议类型(openai_compat:OpenAI兼容;anthropic:Claude原生)",
    )
    auth_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="bearer",
        comment="认证方式(bearer:Authorization Bearer;x-api-key:Anthropic风格;custom:自定义请求头)",
    )
    default_headers: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment='默认请求头(JSON,如{"anthropic-version":"2023-06-01"})'
    )
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="排序序号(数字越小越靠前)"
    )
    health_check_enabled: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="健康检查开关(1:开启,参与熔断判定;0:关闭)"
    )
    remark: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="运维备注(账号归属/合同号/商务信息)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
