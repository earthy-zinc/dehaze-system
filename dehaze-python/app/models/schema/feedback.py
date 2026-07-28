from typing import Optional

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class RatingCreateForm(BaseModel):
    predLogId: int = Field(..., description="处理记录ID")
    rating: int = Field(..., ge=1, le=5, description="评分（1-5）")
    comment: Optional[str] = Field(default=None, max_length=500, description="评价文字")
    tags: Optional[list[str]] = Field(default=None, description="评价标签")
    imageUrls: Optional[list[str]] = Field(default=None, description="截图URL（最多3张）")
    isAnonymous: Optional[int] = Field(default=0, ge=0, le=1, description="是否匿名")


class RatingQuery(BasePageQuery):
    keywords: Optional[str] = None
    algorithmId: Optional[int] = None
    ratingMin: Optional[int] = Field(default=None, ge=1, le=5)
    ratingMax: Optional[int] = Field(default=None, ge=1, le=5)
    hasComment: Optional[bool] = None
    tags: Optional[list[str]] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None


class RatingReplyForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="回复内容")


class FeedbackCreateForm(BaseModel):
    feedbackType: str = Field(..., description="反馈类型")
    title: str = Field(..., min_length=1, max_length=50, description="反馈标题")
    content: str = Field(..., min_length=1, max_length=1000, description="反馈内容")
    contact: Optional[str] = Field(default=None, max_length=64, description="联系方式")
    images: Optional[list[str]] = Field(default=None, description="截图（最多5张）")
    relatedModule: Optional[str] = Field(default=None, max_length=32, description="相关模块")


class FeedbackQuery(BasePageQuery):
    keywords: Optional[str] = None
    feedbackType: Optional[str] = None
    status: Optional[str] = None
    relatedModule: Optional[str] = None
    priority: Optional[int] = None
    assigneeId: Optional[int] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None


class FeedbackSupplementForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="补充内容")
    attachments: Optional[list[str]] = Field(default=None, description="附件URL")


class FeedbackReplyForm(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="回复内容")
    replyType: Optional[str] = Field(default=None, description="回复类型")
    attachments: Optional[list[str]] = Field(default=None, description="附件URL")


class FeedbackAssignForm(BaseModel):
    assigneeId: int = Field(..., description="处理人ID")


class FeedbackCloseForm(BaseModel):
    closeReason: str = Field(..., min_length=1, max_length=256, description="关闭原因")
