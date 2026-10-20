"""
AI 对话模块 - 中间产物 Schema 模型
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.models.schema.common import BasePageQuery, OrmResult


class ArtifactResult(OrmResult):
    id: int = Field(description="主键")
    conversation_id: int = Field(description="会话ID")
    message_id: int = Field(description="关联消息ID")
    type: str = Field(description="产物类型")
    ref_type: str | None = Field(default=None, description="引用业务表")
    ref_id: int | None = Field(default=None, description="引用业务表ID")
    summary: Any | None = Field(default=None, description="业务摘要元数据")
    is_invalid: int = Field(description="引用对象是否已失效")
    create_time: datetime | None = Field(default=None, description="创建时间")


class ArtifactPageQuery(BasePageQuery):
    conversationId: int | None = Field(default=None, description="会话ID")
