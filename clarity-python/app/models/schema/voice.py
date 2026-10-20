"""
语音交互模块 - 热词管理 Schema 模型
"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.schema.common import OrmResult


class HotwordForm(BaseModel):
    word: str = Field(..., min_length=1, max_length=64, description="热词内容")


class HotwordResult(OrmResult):
    id: int = Field(description="主键")
    word: str = Field(description="热词内容")
    create_time: datetime | None = Field(default=None, description="创建时间")
