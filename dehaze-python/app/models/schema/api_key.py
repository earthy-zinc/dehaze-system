from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="密钥名称")
    expiresAt: Optional[datetime] = Field(default=None, description="过期时间")


class ApiKeyResult(BaseModel):
    id: int = Field(description="密钥ID")
    name: str = Field(description="密钥名称")
    apiKey: Optional[str] = Field(default=None, description="密钥明文(仅创建时返回)")
    keyPrefix: str = Field(description="密钥前缀")
    status: int = Field(description="状态(1:正常;0:禁用)")
    expiresAt: Optional[datetime] = Field(default=None, description="过期时间")
    lastUsedAt: Optional[datetime] = Field(default=None, description="最后使用时间")
    createTime: Optional[datetime] = Field(default=None, description="创建时间")
