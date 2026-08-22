from typing import Literal

from pydantic import BaseModel, Field


class RatingCreateForm(BaseModel):
    predLogId: int = Field(..., description="处理记录ID")
    rating: int = Field(..., ge=1, le=5, description="评分（1-5）")
    comment: str | None = Field(default=None, max_length=500, description="评价文字")
    tags: list[str] | None = Field(default=None, description="评价标签")
    imageUrls: list[str] | None = Field(default=None, description="截图URL（最多3张）")
    isAnonymous: int | None = Field(default=0, ge=0, le=1, description="是否匿名")


class RatingReplyForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="回复内容")


class FeedbackCreateForm(BaseModel):
    feedbackType: Literal["suggestion", "bug", "experience", "complaint"] = Field(
        ..., description="反馈类型（suggestion/bug/experience/complaint）"
    )
    title: str = Field(..., min_length=5, max_length=50, description="反馈标题")
    content: str = Field(..., min_length=10, max_length=1000, description="反馈内容")
    contact: str | None = Field(default=None, max_length=64, description="联系方式")
    images: list[str] | None = Field(default=None, description="截图（最多5张）")
    relatedModule: str | None = Field(default=None, max_length=32, description="相关模块")


class FeedbackSupplementForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="补充内容")
    attachments: list[str] | None = Field(default=None, description="附件URL")


class FeedbackReplyForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="回复内容")
    replyType: str | None = Field(default=None, description="回复类型")
    attachments: list[str] | None = Field(default=None, description="附件URL")


class FeedbackAssignForm(BaseModel):
    assigneeId: int = Field(..., description="处理人ID")


class FeedbackCloseForm(BaseModel):
    closeReason: str = Field(..., min_length=1, max_length=256, description="关闭原因")
