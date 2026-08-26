"""外部 MCP Server 管理 Schema 模型（F-M08-006 §2.6.13）。

字段对齐 SDK model.ts（McpToolVO/McpNamespaceVO/McpMarketPresetVO/McpCallVO），
响应字段 snake_case 定义、序列化输出 camelCase（继承 OrmResult），与
McpServerResult 等保持一致；分页查询字段沿用 SDK 分页约定（pageNum/pageSize）。
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.models.schema.common import BasePageQuery, OrmResult


class McpToolResult(OrmResult):
    """MCP 工具清单/详情项（工具名/描述/参数 schema 概要）"""

    name: str = Field(description="工具名(命名空间内唯一)")
    description: str | None = Field(default=None, description="工具描述")
    input_schema: dict[str, Any] | None = Field(default=None, description="参数schema概要")


class McpNamespaceItem(OrmResult):
    """命名空间项（工具分组）：name 为命名空间标识，toolNames 为组内工具名数组"""

    name: str = Field(description="命名空间标识(工具分组)")
    toolNames: list[str] = Field(description="组内工具名数组")


class McpNamespaceUpdate(OrmResult):
    """命名空间配置请求体（覆盖式更新，整组提交）"""

    name: str = Field(description="命名空间标识(工具分组)")
    toolNames: list[str] = Field(default_factory=list, description="组内工具名数组")


class McpMarketPreset(OrmResult):
    """MCP 市场预设目录项（内置静态配置，installed 由同名 Server 推导）"""

    preset_id: str = Field(description="预设ID(市场唯一标识)")
    name: str = Field(description="预设名称")
    description: str | None = Field(default=None, description="预设描述")
    capability_tags: list[str] = Field(default_factory=list, description="能力标签")
    installed: bool = Field(default=False, description="是否已接入(存在同名Server)")


class McpCallResult(OrmResult):
    """外部 MCP 工具调用审计记录（AppendOnly）"""

    id: int = Field(description="主键")
    user_id: int | None = Field(default=None, description="调用用户ID")
    server_id: int = Field(description="关联Server ID")
    server_name: str | None = Field(default=None, description="Server名称(冗余快照)")
    tool_name: str = Field(description="被调用的工具名")
    result: str = Field(description="调用结果(success/failure)")
    latency_ms: int | None = Field(default=None, description="调用耗时(毫秒)")
    create_time: datetime | None = Field(default=None, description="调用时间")


class McpCallQuery(BasePageQuery):
    """外部 MCP 工具调用审计查询参数"""

    server_id: int | None = Field(default=None, description="按Server筛选")
    tool_name: str | None = Field(default=None, description="按工具名筛选")


class McpServerQuery(BasePageQuery):
    """外部 MCP Server 分页查询参数"""

    keyword: str | None = Field(default=None, description="关键字(按名称/描述模糊搜索)")
    status: int | None = Field(default=None, description="状态筛选(1:启用;0:禁用)")


class McpServerCreate(OrmResult):
    """注册外部 MCP Server 表单"""

    name: str = Field(..., min_length=1, max_length=128, description="Server名称(唯一)")
    description: str | None = Field(default=None, max_length=512, description="描述")
    protocol_type: str = Field(
        default="streamable-http", max_length=32, description="传输协议(stdio;streamable-http;sse)"
    )
    endpoint: str | None = Field(default=None, max_length=512, description="端点URL(stdio可为空)")
    auth_type: str | None = Field(default=None, max_length=32, description="鉴权方式(none;api_key;oauth2等)")


class McpServerUpdate(OrmResult):
    """更新外部 MCP Server 配置"""

    name: str | None = Field(default=None, min_length=1, max_length=128, description="Server名称(唯一)")
    description: str | None = Field(default=None, max_length=512, description="描述")
    protocol_type: str | None = Field(default=None, max_length=32, description="传输协议")
    endpoint: str | None = Field(default=None, max_length=512, description="端点URL")
    auth_type: str | None = Field(default=None, max_length=32, description="鉴权方式")


class McpServerStatusForm(OrmResult):
    """启停 Server 表单"""

    status: int = Field(..., ge=0, le=1, description="状态(1:启用;0:禁用)")


class McpServerResult(OrmResult):
    """MCP Server 视图对象（字段 snake_case 定义，序列化输出 camelCase）"""

    id: int = Field(description="主键")
    name: str = Field(description="Server名称")
    description: str | None = Field(default=None, description="描述")
    protocol_type: str = Field(description="传输协议")
    endpoint: str | None = Field(default=None, description="端点URL")
    auth_type: str | None = Field(default=None, description="鉴权方式")
    status: int = Field(description="状态(1:启用;0:禁用)")
    health: str | None = Field(default=None, description="健康状态(online;offline)")
    tool_count: int = Field(default=0, description="工具数量")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class McpHealthResult(OrmResult):
    """Server 健康探测结果"""

    status: str = Field(description="健康状态(online;offline)")
    latency_ms: int | None = Field(default=None, description="延迟(毫秒)")


class McpCredentialForm(OrmResult):
    """外部服务凭据配置表单（加密存储，仅录入/更新，不回显明文）"""

    api_key: str | None = Field(default=None, max_length=1024, description="API Key等外部服务凭据")
    extra: dict[str, str] | None = Field(default=None, description="其他凭据字段(服务层加密存储)")
