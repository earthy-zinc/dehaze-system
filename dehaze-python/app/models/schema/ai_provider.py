"""
AI 模型供应商管理 Schema 模型
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.models.schema.common import OrmResult

# ── 供应商管理 ──────────────────────────────────────────


class ProviderCreate(OrmResult):
    provider_code: str = Field(..., min_length=1, max_length=32, description="供应商编码")
    display_name: str = Field(..., min_length=1, max_length=128, description="显示名称")
    api_base_url: str = Field(..., min_length=1, max_length=512, description="API基础地址")
    protocol_type: str = Field(default="openai_compat", max_length=32, description="协议类型")
    auth_type: str = Field(default="bearer", max_length=32, description="认证方式")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    sort_order: int = Field(default=0, description="排序序号")
    health_check_enabled: int = Field(
        default=1, description="健康检查开关(1:开启,参与熔断判定;0:关闭)"
    )
    remark: str | None = Field(default=None, description="运维备注")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class ProviderUpdate(OrmResult):
    display_name: str | None = Field(
        default=None, min_length=1, max_length=128, description="显示名称"
    )
    api_base_url: str | None = Field(
        default=None, min_length=1, max_length=512, description="API基础地址"
    )
    protocol_type: str | None = Field(default=None, max_length=32, description="协议类型")
    auth_type: str | None = Field(default=None, max_length=32, description="认证方式")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    sort_order: int | None = Field(default=None, description="排序序号")
    health_check_enabled: int | None = Field(
        default=None, description="健康检查开关(1:开启;0:关闭)"
    )
    remark: str | None = Field(default=None, description="运维备注")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")


class ProviderResult(OrmResult):
    id: int = Field(description="主键")
    provider_code: str = Field(description="供应商编码")
    display_name: str = Field(description="显示名称")
    api_base_url: str = Field(description="API基础地址")
    protocol_type: str = Field(description="协议类型")
    auth_type: str = Field(description="认证方式")
    default_headers: dict[str, Any] | None = Field(default=None, description="默认请求头(JSON)")
    sort_order: int = Field(description="排序序号")
    health_check_enabled: int = Field(
        default=1, description="健康检查开关(1:开启,参与熔断判定;0:关闭)"
    )
    remark: str | None = Field(default=None, description="运维备注")
    health: str | None = Field(
        default=None, description="健康状态(healthy:健康;suspicious:可疑;open:熔断)"
    )
    status: int = Field(description="状态(1:启用;0:禁用)")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


# ── API Key 管理 ──────────────────────────────────────────


class ProviderKeyCreate(OrmResult):
    name: str = Field(..., min_length=1, max_length=128, description="Key名称")
    key: str = Field(..., min_length=1, description="Key明文(service层加密后不存储此字段)")
    priority: int = Field(default=0, description="优先级(数字越小越优先)")
    weight: int = Field(default=1, ge=1, description="权重")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class ProviderKeyUpdate(OrmResult):
    name: str | None = Field(default=None, min_length=1, max_length=128, description="Key名称")
    priority: int | None = Field(default=None, description="优先级")
    weight: int | None = Field(default=None, ge=1, description="权重")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    expires_at: datetime | None = Field(default=None, description="过期时间")


class ProviderKeyResult(OrmResult):
    id: int = Field(description="主键")
    provider_id: int = Field(description="关联供应商ID")
    name: str = Field(description="Key名称")
    key_prefix: str | None = Field(default=None, description="密钥前缀(展示用)")
    status: int = Field(description="状态(1:启用;0:禁用)")
    priority: int = Field(description="优先级")
    weight: int = Field(description="权重")
    daily_quota: int | None = Field(default=None, description="日调用上限")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    last_used_at: datetime | None = Field(default=None, description="最后使用时间")
    last_used_by: int | None = Field(default=None, description="最后使用的用户ID")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")
