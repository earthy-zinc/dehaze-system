from typing import Any

from pydantic import BaseModel, Field


class MessageSendRequest(BaseModel):
    templateCode: str | None = None
    type: str = Field(..., min_length=1, description="消息类型")
    title: str | None = None
    content: str | None = None
    recipientIds: list[int] = Field(..., min_length=1, description="接收人ID列表")
    bizModule: str | None = None
    bizId: str | None = None
    priority: int | None = Field(default=None, ge=1, le=4)
    jumpUrl: str | None = None
    variables: dict[str, str] | None = None
    extra: dict[str, Any] | None = None


class NotificationSettingsForm(BaseModel):
    pushEnabled: bool | None = None
    dndEnabled: bool | None = None
    dndStart: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
        description="免打扰开始时间(HH:mm:ss)",
    )
    dndEnd: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
        description="免打扰结束时间(HH:mm:ss)",
    )
    preferences: dict[str, Any] | None = None


class AnnouncementForm(BaseModel):
    title: str = Field(..., min_length=2, max_length=50, description="公告标题(2-50字符)")
    content: str = Field(..., min_length=1, description="公告内容")
    type: str = Field(..., min_length=1, description="公告类型")
    importance: int = Field(..., ge=1, le=2, description="重要级别(1:普通;2:重要)")
    targetScope: str = Field(..., min_length=1, description="发送范围")
    targetParams: dict[str, Any] | None = None
    sendTime: str | None = None
    expireTime: str | None = None


class AnnouncementUpdateForm(BaseModel):
    title: str | None = Field(
        default=None, min_length=2, max_length=50, description="公告标题(2-50字符)"
    )
    content: str | None = Field(default=None, min_length=1, description="公告内容")
    type: str | None = Field(default=None, min_length=1, description="公告类型")
    importance: int | None = Field(default=None, ge=1, le=2, description="重要级别(1:普通;2:重要)")
    targetScope: str | None = Field(default=None, min_length=1, description="发送范围")
    targetParams: dict[str, Any] | None = None
    sendTime: str | None = None
    expireTime: str | None = None


class MessageTemplateForm(BaseModel):
    name: str | None = None
    titleTemplate: str | None = None
    contentTemplate: str | None = None
    priority: int | None = Field(default=None, ge=1, le=4)
    channels: dict[str, bool] | None = None
    status: int | None = Field(default=None, ge=0, le=1)
