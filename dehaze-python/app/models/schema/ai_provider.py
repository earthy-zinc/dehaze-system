"""
AI 模型供应商管理 Schema 模型
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.models.schema.common import OrmResult


class UserIdentityForwardConfig(OrmResult):
    """用户身份透传配置（抽象覆盖 DeepSeek user_id/OpenAI user/Anthropic metadata.user_id）。"""

    enabled: bool = Field(description="是否启用透传")
    field: str = Field(description="透传字段名或嵌套路径(user_id/user/metadata.user_id)")
    prefix: str | None = Field(default=None, description="透传值脱敏前缀")
    max_len: int | None = Field(default=None, description="透传值最大长度")


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
    user_identity_forward: UserIdentityForwardConfig | None = Field(
        default=None, description="用户身份透传配置"
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
    user_identity_forward: UserIdentityForwardConfig | None = Field(
        default=None, description="用户身份透传配置"
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
    user_identity_forward: UserIdentityForwardConfig | None = Field(
        default=None, description="用户身份透传配置"
    )
    remark: str | None = Field(default=None, description="运维备注")
    health: str | None = Field(
        default=None, description="健康状态(healthy:健康;suspicious:可疑;open:熔断)"
    )
    status: int = Field(description="状态(1:启用;0:禁用)")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class ProviderKeyCreate(OrmResult):
    name: str = Field(..., min_length=1, max_length=128, description="Key名称")
    key: str = Field(..., min_length=1, description="Key明文(service层加密后不存储此字段)")
    priority: int = Field(default=0, description="优先级(数字越小越优先)")
    weight: int = Field(default=1, ge=1, description="权重")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    rpm_limit: int | None = Field(default=None, ge=0, description="分钟调用频率上限(0=不限)")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")


class ProviderKeyUpdate(OrmResult):
    name: str | None = Field(default=None, min_length=1, max_length=128, description="Key名称")
    priority: int | None = Field(default=None, description="优先级")
    weight: int | None = Field(default=None, ge=1, description="权重")
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")
    daily_quota: int | None = Field(default=None, ge=1, description="日调用上限")
    rpm_limit: int | None = Field(default=None, ge=0, description="分钟调用频率上限(0=不限)")
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
    rpm_limit: int | None = Field(default=None, description="分钟调用频率上限(0=不限)")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    last_used_at: datetime | None = Field(default=None, description="最后使用时间")
    last_used_by: int | None = Field(default=None, description="最后使用的用户ID")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


# ==================== 运营统计（管理端，GET /api/v1/ai/usage/stats） ====================


class UsageStatsQuery(OrmResult):
    start_time: datetime | None = Field(default=None, description="开始时间")
    end_time: datetime | None = Field(default=None, description="结束时间")
    granularity: str = Field(default="day", description="聚合粒度(day:按日;month:按月)")


class ProviderHealthStatResult(OrmResult):
    provider_id: int = Field(description="供应商ID")
    provider_name: str = Field(description="供应商显示名称")
    health: str = Field(description="健康状态(healthy/suspicious/open)")
    call_count: int = Field(description="调用次数")
    success_rate: float = Field(description="成功率(0-1)")
    rate429: float = Field(description="429限流率(0-1)")
    p95_latency_ms: int = Field(default=0, description="延迟P95(毫秒)")
    circuit_open: bool = Field(description="是否熔断")


class ModelUsageStatResult(OrmResult):
    model_id: str = Field(description="模型标识")
    display_name: str = Field(description="模型显示名称")
    call_count: int = Field(description="调用次数")
    input_tokens: int = Field(description="输入Token数")
    output_tokens: int = Field(description="输出Token数")
    credits: int = Field(description="积分开销")


class DowngradeStatResult(OrmResult):
    model_id: str = Field(description="被降级的原选模型标识")
    count: int = Field(description="降级次数")


class DegradeFaultStatResult(OrmResult):
    downgrade_frequency: list[DowngradeStatResult] = Field(default_factory=list, description="各模型降级频率")
    key_failover_count: int = Field(default=0, description="Key失败切换次数")


class UsageStatsResult(OrmResult):
    provider_health: list[ProviderHealthStatResult] = Field(
        default_factory=list, description="供应商健康看板"
    )
    model_usage: list[ModelUsageStatResult] = Field(
        default_factory=list, description="模型用量分布"
    )
    degrade_fault: DegradeFaultStatResult = Field(
        default_factory=DegradeFaultStatResult, description="降级与故障统计"
    )
