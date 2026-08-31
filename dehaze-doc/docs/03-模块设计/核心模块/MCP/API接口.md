# MCP 能力网关模块 API 接口

## 1. 文档概述

本文档定义**内部 MCP 能力网关**（dehaze-mcp-gateway，系统能力出口）的接口规范。它与"MCP Server 管理"（外部能力进口，属 AI 对话模块能力扩展域）是两条独立通道，概念界定见 [需求规格.md](需求规格.md) §1.0。

- **传输协议**：Streamable HTTP（`POST /mcp` 单一端点）
- **协议规范**：JSON-RPC 2.0 over HTTP（遵循 MCP 2025 规范）
- **认证**：API Key 通过请求头 `x-dehaze-api-key` 或环境变量传递，透传到后端，后端负责校验

## 2. 元 tool 接口

MCP 工具暴露 3 个元 tool，LLM 通过它们按需发现和调用后端 API。

### 2.1 lookup_tool

搜索后端 API，返回工具名、描述、参数名列表（`*` 标必填）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词（中文按双字 bigram 分词） |

返回示例：

```
post_api_v1_prediction: 执行模型预测（去雾处理，异步） | 参数: algorithmId*, fileId, imageUrl, params, recommendedBy
get_api_v1_prediction_logs: 获取预测日志列表 | 参数: pageNum, pageSize, algorithmId
```

无匹配时返回相近候选或提示调整关键词。

### 2.2 lookup_tool_param_schema

查看指定 API 的完整参数定义。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tool_name` | string | 是 | 工具名（来自 lookup_tool） |

返回结构化 JSON：

| 字段 | 说明 |
|------|------|
| `description` | API 描述 |
| `method` / `path` | HTTP 方法和路径 |
| `namespace` | 命名空间（OpenAPI tag） |
| `params[]` | 每个参数：name / location / required / schema 描述 / example |
| `arguments_example` | 必填参数的完整示例（嵌套结构可直接用） |

### 2.3 execute_tool

调用指定 API。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tool_name` | string | 是 | 工具名（来自 lookup_tool） |
| `arguments` | string | 否 | JSON 字符串，默认 `{}` |

本地校验：必填参数缺失时返回"缺少必填参数: xxx"。

### 2.4 响应转换

| 后端响应 | MCP 响应 |
|------|------|
| HTTP 200 | 返回响应体 |
| HTTP 200 + 超长（>8000 字符） | `响应过长已截断（共N字符，仅展示前8000）：\n{截断内容}` |
| HTTP 4xx | `错误({状态码}): {响应体}` |
| HTTP 5xx / 超时 | `后端服务不可用({状态码}): {响应体中的诊断信息}` |

## 3. 错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `-32700` | Parse error | JSON 格式不合法 |
| `-32600` | Invalid Request | 请求格式不合法 |
| `-32601` | Method not found | 调用的方法不存在 |
| `-32602` | Invalid params | 参数校验失败 |
| `-32603` | Internal error | 内部错误 |

> 后端业务错误（HTTP 400/401/403/404/429）不使用 JSON-RPC error，而是通过 `content + isError` 返回，让 LLM 据此决策。认证、鉴权、限流、配额均由后端处理。
