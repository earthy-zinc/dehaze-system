"""
AI 对话模块 - 长期记忆 Schema 模型
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery, OrmResult


class MemoryResult(OrmResult):
    id: int = Field(description="主键")
    user_id: int = Field(description="用户ID")
    memory_type: str = Field(description="记忆类型(episodic/semantic/procedural)")
    content: str = Field(description="记忆内容")
    metadata_: Any | None = Field(
        default=None, serialization_alias="metadata", description="结构化属性"
    )
    importance: int = Field(description="重要性评分")
    access_count: int = Field(description="检索命中次数")
    last_accessed_at: datetime | None = Field(default=None, description="最后访问时间")
    source: str = Field(description="来源")
    status: int = Field(description="状态(1:启用;0:禁用)")
    archived: int = Field(description="是否归档")
    create_time: datetime = Field(description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class MemoryCreate(BaseModel):
    memoryType: str = Field(..., description="记忆类型(episodic/semantic/procedural)")
    content: str = Field(..., max_length=2000, description="记忆内容")
    metadata: dict | None = Field(default=None, description="结构化属性")
    importance: int = Field(default=50, ge=0, le=100, description="重要性评分")
    source: str = Field(default="manual", description="来源")


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, max_length=2000, description="记忆内容")
    importance: int | None = Field(default=None, ge=0, le=100, description="重要性评分")
    status: int | None = Field(default=None, ge=0, le=1, description="状态(1:启用;0:禁用)")


class MemoryPageQuery(BasePageQuery):
    memoryType: str | None = Field(default=None, description="记忆类型过滤")
    source: str | None = Field(
        default=None, description="来源过滤(conversation/feedback/reflection/manual)"
    )


class MemoryClearQuery(BaseModel):
    memoryType: str | None = Field(default=None, description="清空指定类型(为空则全部)")
    start: datetime | None = Field(default=None, description="时间范围起(按创建时间)")
    end: datetime | None = Field(default=None, description="时间范围止(按创建时间)")
