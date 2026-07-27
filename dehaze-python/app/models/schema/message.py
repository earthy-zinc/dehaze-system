from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageQuery(BaseModel):
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=20, ge=1, le=100, description="每页条数")
    type: Optional[str] = Field(default=None, description="消息类型")
    readStatus: Optional[int] = Field(default=None, ge=0, le=1, description="已读状态")


class MessageSearchQuery(BaseModel):
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=20, ge=1, le=100, description="每页条数")
    keyword: str = Field(..., min_length=1, description="搜索关键字")


class MessageVO(BaseModel):
    id: int
    type: str
    typeLabel: str
    title: str
    summary: Optional[str] = None
    priority: int
    readStatus: int
    senderType: int
    jumpUrl: Optional[str] = None
    createTime: Optional[str] = None


class MessageDetailVO(BaseModel):
    id: int
    type: str
    typeLabel: str
    title: str
    content: str
    priority: int
    senderType: int
    senderTypeLabel: str
    readStatus: int
    readTime: Optional[str] = None
    jumpUrl: Optional[str] = None
    extra: Optional[Any] = None
    createTime: Optional[str] = None


class UnreadCountVO(BaseModel):
    count: int


class ReadAllResult(BaseModel):
    affectedCount: int


class MessageSendRequest(BaseModel):
    templateCode: Optional[str] = None
    type: str = Field(..., min_length=1, description="消息类型")
    title: Optional[str] = None
    content: Optional[str] = None
    recipientIds: list[int] = Field(..., min_length=1, description="接收人ID列表")
    bizModule: Optional[str] = None
    bizId: Optional[str] = None
    priority: Optional[int] = Field(default=2, ge=1, le=4)
    jumpUrl: Optional[str] = None
    variables: Optional[dict[str, str]] = None
    extra: Optional[dict[str, Any]] = None


class MessageSendResult(BaseModel):
    messageIds: list[int]


class NotificationSettings(BaseModel):
    pushEnabled: bool
    dndEnabled: bool
    dndStart: Optional[str] = None
    dndEnd: Optional[str] = None
    preferences: Optional[dict[str, Any]] = None


class NotificationSettingsForm(BaseModel):
    pushEnabled: Optional[bool] = None
    dndEnabled: Optional[bool] = None
    dndStart: Optional[str] = None
    dndEnd: Optional[str] = None
    preferences: Optional[dict[str, Any]] = None


class AnnouncementQuery(BaseModel):
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=10, ge=1, le=100, description="每页条数")
    title: Optional[str] = None
    type: Optional[str] = None
    status: Optional[int] = Field(default=None, ge=1, le=4)


class AnnouncementVO(BaseModel):
    id: int
    title: str
    content: Optional[str] = None
    type: str
    typeLabel: Optional[str] = None
    importance: int
    importanceLabel: Optional[str] = None
    targetScope: str
    targetScopeLabel: Optional[str] = None
    targetParams: Optional[Any] = None
    status: int
    statusLabel: Optional[str] = None
    sendTime: Optional[str] = None
    expireTime: Optional[str] = None
    sentCount: Optional[int] = None
    createTime: Optional[str] = None
    updateTime: Optional[str] = None
    createBy: Optional[int] = None


class AnnouncementForm(BaseModel):
    title: str = Field(..., min_length=2, max_length=50, description="公告标题(2-50字符)")
    content: str = Field(..., min_length=1, description="公告内容")
    type: str = Field(..., min_length=1, description="公告类型")
    importance: int = Field(..., ge=1, le=2, description="重要级别(1:普通;2:重要)")
    targetScope: str = Field(..., min_length=1, description="发送范围")
    targetParams: Optional[dict[str, Any]] = None
    sendTime: Optional[str] = None
    expireTime: Optional[str] = None


class AnnouncementSendResult(BaseModel):
    sentCount: int


class MessageTemplateQuery(BaseModel):
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=20, ge=1, le=100, description="每页条数")
    name: Optional[str] = None
    type: Optional[str] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)


class MessageTemplateVO(BaseModel):
    id: int
    code: str
    name: str
    type: str
    titleTemplate: str
    contentTemplate: Optional[str] = None
    priority: int
    channels: Optional[dict[str, bool]] = None
    variables: Optional[list[dict[str, str]]] = None
    status: int
    createTime: Optional[str] = None
    updateTime: Optional[str] = None


class MessageTemplateForm(BaseModel):
    name: Optional[str] = None
    titleTemplate: Optional[str] = None
    contentTemplate: Optional[str] = None
    priority: Optional[int] = Field(default=None, ge=1, le=4)
    channels: Optional[dict[str, bool]] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)
