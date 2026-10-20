from datetime import datetime

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="密钥名称")
    expiresAt: datetime | None = Field(default=None, description="过期时间")
    dailyQuota: int | None = Field(default=None, ge=1, description="日调用配额(不限制则不传或传0)")
    monthlyQuota: int | None = Field(
        default=None, ge=1, description="月调用配额(不限制则不传或传0)"
    )
    rpmLimit: int | None = Field(
        default=None, ge=1, description="每分钟请求数上限RPM(不限制则不传或传0)"
    )
    modelWhitelist: list[str] | None = Field(
        default=None, description="模型白名单(不限制或继承用户可见模型则不传或传空数组)"
    )


class ApiKeyResult(BaseModel):
    id: int = Field(description="密钥ID")
    name: str = Field(description="密钥名称")
    apiKey: str | None = Field(default=None, description="密钥明文(仅创建时返回)")
    keyPrefix: str = Field(description="密钥前缀")
    status: int = Field(description="状态(1:正常;0:禁用)")
    expiresAt: datetime | None = Field(default=None, description="过期时间")
    lastUsedAt: datetime | None = Field(default=None, description="最后使用时间")
    createTime: datetime | None = Field(default=None, description="创建时间")
    dailyQuota: int | None = Field(default=None, description="日调用配额(NULL表示不限制)")
    monthlyQuota: int | None = Field(default=None, description="月调用配额(NULL表示不限制)")
    rpmLimit: int | None = Field(default=None, description="每分钟请求数上限RPM(NULL表示不限制)")
    modelWhitelist: list[str] | None = Field(
        default=None, description="模型白名单(NULL表示继承用户可见模型)"
    )
