# MCP 能力网关模块 API 接口

## 1. 文档概述

本文档定义 **MCP 能力网关** 模块的接口规范，是该模块 API 契约的**唯一权威来源**。

- **传输协议**：Streamable HTTP（`POST /mcp` 单一端点）
- **协议规范**：JSON-RPC 2.0 over HTTP（遵循 MCP 2025 规范）
- **消息格式**：JSON-RPC 2.0（`jsonrpc:"2.0"`，含 `id`/`method`/`params`/`result` 标准字段）
- **认证**：Bearer Token（`Authorization: Bearer dhak_xxx`）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

> 接口详细参数/响应结构可通过 MCP 协议交互或 API 文档 MCP 查询，本文档仅定义方法清单和权限标识。

## 2. 接口清单

### 2.1 MCP 标准方法（JSON-RPC over `/mcp`）

| 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|---------|---------|-----------|
| `initialize` | 能力协商，服务端返回支持的 capabilities | - | F-M07-001 |
| `tools/list` | 返回当前用户可见的工具列表（按 VIP 等级过滤）。默认仅返回命名空间摘要（name + description + tool_count），传 `expand` 参数时展开为完整工具定义，避免上百个工具定义撑爆 LLM 上下文 | - | F-M07-002 |
| `tools/search` | 网关扩展方法。按自然语言关键字搜索匹配的工具，返回完整定义，用于按需加载 | - | F-M07-002 |
| `tools/call` | 调用指定工具 | - | F-M07-003 |
| `resources/list` | 返回所有可用 resource URI | - | F-M07-004 |
| `resources/read` | 按 URI 读取 resource 内容 | - | F-M07-004 |
| `prompts/list` | 返回所有预定义 prompt 模板 | - | F-M07-005 |
| `prompts/get` | 获取指定 prompt 模板的完整内容 | - | F-M07-005 |
| `notifications/tools/list_changed` | 网关主动推送，通知客户端工具列表已变更（Server → Client，客户端收到后应调用 `tools/list` 刷新） | - | F-M07-002 |

**tools/search 参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 自然语言搜索查询，描述需要什么能力的工具 |
| `detail` | string | 否 | 返回详细级别：`name_only`/`summary`/`full`，默认 `full` |
| `limit` | integer | 否 | 最大返回数，默认 5，范围 [1, 10] |

> **tools/call 异步调用**：后端返回 `task_id` 时，AI 对话模块进入 `async_wait` 中断，等待任务管理模块/算法服务回调恢复。后端业务错误（HTTP 400/401/403/404/429）不使用 JSON-RPC error，而是通过 `content + isError: true` 返回（工具调用已成功派发，后端返回业务错误）。

### 2.2 管理接口（REST）

> 后台管理（F-M07-008）的 REST 端点，区别于上述 MCP 协议方法（JSON-RPC over `/mcp`）。用于工具注册表维护与过滤规则在线配置。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/admin/mcp/refresh` | POST | 手动刷新：立即重新读取 OpenAPI 并更新注册表，返回 diff（新增/删除/变更） | `mcp:tool:manage` | F-M07-008 |
| `/admin/mcp/tools` | GET | 查询当前已注册的 tool/resource/prompt 列表（含 name、description、对应后端 API 路径、命名空间归属） | `mcp:tool:manage` | F-M07-008 |
| `/admin/mcp/filter-config` | GET | 查询过滤规则配置（路径白名单、标签过滤、显式排除列表等） | `mcp:tool:manage` | F-M07-008 |
| `/admin/mcp/filter-config` | PUT | 保存过滤规则配置，保存后触发即时刷新（无需重启网关） | `mcp:tool:manage` | F-M07-008 |

> `POST /admin/mcp/refresh` 无变更时 `diff` 为空且不推送 `notifications/tools/list_changed`。

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | MCP 标准方法（initialize/tools/resources/prompts）仅需 Bearer Token 认证，VIP 等级控制可见工具范围 |
| `mcp:tool:manage` | 工具注册表管理（刷新、查询、过滤规则配置），仅管理员 |

## 4. 业务错误码

JSON-RPC 协议级错误使用标准 JSON-RPC error code；后端业务错误通过 `content + isError: true` 返回（见 §2.1 tools/call 异步调用说明）。

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `-32700` | Parse error | JSON 格式不合法 |
| `-32600` | Invalid Request | 请求格式不合法 |
| `-32601` | Method not found | 调用的方法不存在 |
| `-32602` | Invalid params | 参数校验失败 |
| `-32603` | Internal error | 网关内部错误 |
| `-32001` | Authentication failed | Bearer Token 无效/过期/已吊销 |
| `-32002` | Tool not found | 调用的 tool 不存在或用户无权访问 |
| `-32003` | Permission denied | VIP 等级不足，无权访问该 tool |
| `-32004` | Rate limited | 网关层工具调用频率超限 |
| `-32005` | Backend unavailable | 后端不可用/超时/5xx |
