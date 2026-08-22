"""
AI 对话模块 - 消息反馈 Schema 模型
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.schema.common import OrmResult


class FeedbackResult(OrmResult):
    id: int = Field(description="主键")
    message_id: int = Field(description="消息ID")
    user_id: int = Field(description="用户ID")
    rating: int = Field(description="评分(1:点赞;-1:点踩)")
    tags: Any | None = Field(default=None, description="预设标签(JSON数组)")
    comment: str | None = Field(default=None, description="反馈内容")
    create_time: datetime = Field(description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class FeedbackCreateRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1, description="评分(1:点赞;-1:点踩)")
    tags: list[str] | None = Field(default=None, description="预设标签")
    comment: str | None = Field(default=None, max_length=2000, description="反馈内容")
