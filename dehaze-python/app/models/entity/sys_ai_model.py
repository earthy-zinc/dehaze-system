from typing import Any

from sqlalchemy import JSON, BigInteger, Integer, Numeric, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiModel(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_model"
    __table_args__ = {"comment": "AI模型配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_id: Mapped[int] = mapped_column(
        BigInteger, index=True, nullable=False, comment="关联供应商ID(关联sys_ai_provider.id)"
    )
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="模型标识(如gpt-4o;claude-3-5-sonnet;deepseek-chat)"
    )
    model_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="chat", comment="模型类型(chat:对话;embedding:向量;rerank:重排)"
    )
    dimension: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="embedding向量维度(model_type=embedding时必填;创建后不可改)"
    )
    display_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="显示名称")
    max_context_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=4096, comment="最大上下文Token数"
    )
    max_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=4096, comment="最大输出Token数"
    )
    supports_multimodal: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否支持多模态(0:否;1:是)"
    )
    supports_tool_call: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否支持工具调用(0:否;1:是)"
    )
    supports_streaming: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="是否支持流式输出(0:否;1:是)"
    )
    supports_prompt_cache: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否支持Prompt缓存(0:否;1:是)"
    )
    supports_structured_output: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否支持结构化输出(0:否;1:是)"
    )
    extra_request_params: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="厂商私有请求参数(如阿里云enable_thinking/reasoning_effort)，随请求体透传，核心键不可覆盖",
    )
    fallback_model_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="降级模型ID(关联sys_ai_model.id)"
    )
    prompt_cache_prefix_len: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Prompt缓存稳定前缀长度"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
    vip_level: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="最低可用VIP等级(0:所有用户;1:VIP1及以上;2:VIP2及以上)",
    )
