# MCP 能力网关模块 API 接口

## 1. 文档概述

本文档定义 **MCP 能力网关** 的标准 MCP 协议接口规范。

- **传输协议**：Streamable HTTP（`POST /mcp` 单一端点）
- **消息格式**：JSON-RPC 2.0（`jsonrpc:"2.0"`，含 `id`/`method`/`params`/`result` 标准字段）
- **规范版本**：MCP 2025 规范
- **认证**：Bearer Token（`Authorization: Bearer dhak_xxx`）

## 2. 标准方法

### 2.1 initialize

**功能**：能力协商，服务端返回支持的 capabilities。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "clientInfo": {"name": "dehaze-ai-chat", "version": "1.0.0"}
  }
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {"listChanged": true},
      "resources": {"subscribe": true, "listChanged": true},
      "prompts": {"listChanged": true}
    },
    "serverInfo": {"name": "dehaze-mcp-gateway", "version": "1.0.0"}
  }
}
```

> `listChanged: true` 表示服务端支持主动推送变更通知。

### 2.2 tools/list

**功能**：返回当前用户可见的工具列表（按 VIP 等级过滤）。**默认仅返回命名空间摘要**（每组仅 name + description + tool_count），传递 `expand` 参数时展开为完整工具定义，避免上百个工具的定义撑爆 LLM 上下文。

**请求（命名空间摘要模式，默认）**：
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
```

**响应（命名空间摘要）**：
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "namespaces": [
      {"name": "image_processing", "description": "图像处理工具：去雾、增强、超分辨率等", "tool_count": 6},
      {"name": "evaluation", "description": "图像质量评估工具：PSNR/SSIM/主观评价", "tool_count": 4},
      {"name": "algorithm", "description": "算法管理工具（查询类为 resource）", "tool_count": 2},
      {"name": "knowledge", "description": "知识库检索工具", "tool_count": 2}
    ],
    "total_tools": 14
  }
}
```

**请求（展开指定命名空间）**：
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {"expand": ["image_processing", "algorithm"]}
}
```

**响应（展开后）**：
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "predict_dehaze",
        "namespace": "image_processing",
        "description": "对上传的图片进行去雾处理。支持多种去雾算法，返回处理后的图片。",
        "inputSchema": {
          "type": "object",
          "properties": {
            "fileId": {"type": "integer", "description": "需要处理的图片文件ID"},
            "algorithm": {"type": "string", "description": "去雾算法名称，如 RIDCP/FFA-Net/DCP"},
            "strength": {"type": "integer", "description": "处理强度 (0-100)", "default": 70}
          },
          "required": ["fileId"]
        }
      }
    ]
  }
}
```

> **推荐用法**：AI 对话模块的 before_model 钩子先调用 `tools/list`（无 expand）获取命名空间摘要（~80 tokens），根据用户意图仅展开 1-2 个匹配的命名空间（如用户说"去雾"→ 展开 `image_processing`），减少 93% 上下文占用。

### 2.3 tools/search（网关扩展方法）

> **非 MCP 标准方法**，是网关扩展方法。MCP 标准方法仅含 `initialize`/`tools/list`/`tools/call`/`resources/*`/`prompts/*`/`notifications/*`。`tools/search` 是网关为支持按需加载而扩展的方法，AI 对话模块通过 JSON-RPC `method: "tools/search"` 调用。

**功能**：LLM 按自然语言关键字搜索匹配的工具，返回匹配工具的**完整定义**。用于按需加载——LLM 不预先知道所有工具的完整 definition，只在需要时搜索并加载。参考 OpenAI Agents SDK `ToolSearchTool` 和 Anthropic `search_tools` 设计。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "method": "tools/search",
  "params": {
    "query": "评估图像质量 指标 PSNR",
    "detail": "full"
  }
}
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 自然语言搜索查询，描述需要什么能力的工具 |
| `detail` | string | 否 | 返回详细级别：`name_only`/`summary`/`full`，默认 `full` |
| `limit` | integer | 否 | 最大返回数，默认 5，范围 [1, 10] |

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "result": {
    "matched": 3,
    "total_tools": 14,
    "tools": [
      {
        "name": "evaluate_image",
        "namespace": "evaluation",
        "description": "对处理后的图像进行指标评估，返回 PSNR/SSIM/LPIPS/NIQE 等指标",
        "inputSchema": {}
      },
      {
        "name": "compare_algorithms",
        "namespace": "evaluation",
        "description": "比较多个算法在同一图像上的评估指标",
        "inputSchema": {}
      }
    ]
  }
}
```

**搜索实现**：对工具名称（name）和描述（description）做语义匹配（ES BM25 + 关键词，不依赖 LLM），同命名空间匹配工具一并返回。

### 2.4 tools/call

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "predict_dehaze",
    "arguments": {"fileId": 123, "algorithm": "RIDCP", "strength": 80}
  }
}
```

**成功响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{"type": "text", "text": "处理成功。PSNR: 28.5, SSIM: 0.92"}],
    "isError": false
  }
}
```

**调用错误响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{"type": "text", "text": "配额不足：今日处理次数已用完 (100/100)。请升级VIP或等待明日重置。"}],
    "isError": true
  }
}
```

**JSON-RPC 错误响应**（网关层错误，如参数校验失败）：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "参数 'strength' 超出范围 [0,100]: 150"
  }
}
```

> **异步调用**：后端返回 `task_id` 时，网关透传 `{"content":[{"type":"text","text":"task_id: xxx.operating."}],"isError":false}`，AI 对话模块持久化 checkpoint（task_id → 会话映射）后进入 async_wait 中断，等待任务管理模块/算法服务回调恢复。

### 2.5 resources/list

**功能**：返回所有可用 resource URI。

**请求**：
```json
{"jsonrpc": "2.0", "id": 4, "method": "resources/list", "params": {}}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "resources": [
      {"uri": "algo://list", "name": "算法列表", "description": "系统中所有可用去雾算法清单", "mimeType": "application/json"},
      {"uri": "algo://{id}", "name": "算法详情", "description": "指定算法的详细说明、参数、适用场景"},
      {"uri": "algo://{id}/docs", "name": "算法文档", "description": "算法的技术文档和使用说明"}
    ]
  }
}
```

### 2.6 resources/read

**功能**：按 URI 读取 resource 内容。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/read",
  "params": {"uri": "algo://15"}
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "contents": [{
      "uri": "algo://15",
      "mimeType": "application/json",
      "text": "{\"name\":\"RIDCP\",\"type\":\"dehaze\",\"description\":\"基于颜色衰减先验的去雾算法...\",\"params\":[{\"name\":\"strength\",\"range\":[0,100]}]}"
    }]
  }
}
```

### 2.6 prompts/list

**功能**：返回所有预定义 prompt 模板。

**请求**：
```json
{"jsonrpc": "2.0", "id": 6, "method": "prompts/list", "params": {}}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "prompts": [
      {
        "name": "dehaze-evaluation",
        "description": "去雾效果评估报告模板",
        "arguments": [{"name": "algorithm", "description": "使用的算法名称", "required": true}]
      }
    ]
  }
}
```

### 2.8 prompts/get

**功能**：获取指定 prompt 模板的完整内容。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "prompts/get",
  "params": {"name": "dehaze-evaluation", "arguments": {"algorithm": "RIDCP"}}
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "description": "去雾效果评估报告模板",
    "messages": [{
      "role": "user",
      "content": {
        "type": "text",
        "text": "请按以下标准评估 RIDCP 算法的去雾效果：\n1. PSNR 和 SSIM 指标\n2. 主观视觉效果（是否有光晕/色彩失真）\n3. 处理速度"
      }
    }]
  }
}
```

### 2.9 notifications/tools/list_changed

**功能**：网关主动推送，通知客户端工具列表已变更。

**请求**（Server → Client）：
```json
{"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}}
```

> 客户端收到后应调用 `tools/list` 刷新工具列表。

---

## 3. 管理接口（REST）

> 以下为后台管理（F-M07-008）的 REST 端点，区别于上述 MCP 协议方法（JSON-RPC over `/mcp`）。管理接口需管理员角色，用于工具注册表维护与过滤规则在线配置。实现详见 [后端实现-工具注册.md](./后端实现-工具注册.md) §2.3-§2.4。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/admin/mcp/refresh` | POST | 手动刷新：立即重新读取 OpenAPI 并更新注册表，返回 diff（新增/删除/变更） | 管理员 | F-M07-008 |
| `/admin/mcp/tools` | GET | 查询当前已注册的 tool/resource/prompt 列表（含 name、description、对应后端 API 路径、命名空间归属） | 管理员 | F-M07-008 |
| `/admin/mcp/filter-config` | GET | 查询过滤规则配置（路径白名单、标签过滤、显式排除列表等） | 管理员 | F-M07-008 |
| `/admin/mcp/filter-config` | PUT | 保存过滤规则配置，保存后触发即时刷新（无需重启网关） | 管理员 | F-M07-008 |

**`POST /admin/mcp/refresh` 响应**：返回刷新结果，含 `toolsCount`、`resourcesCount`、`promptsCount`、`diff`（added/removed/changed 列表）。无变更时 `diff` 为空且不推送 `notifications/tools/list_changed`。

**过滤规则配置**：`PUT /admin/mcp/filter-config` 保存后通过 Redis Pub/Sub 广播至多实例，各实例重新加载规则并刷新（多实例一致性策略详见 [后端实现-架构与公共.md](./后端实现-架构与公共.md)）。

---

## 4. JSON-RPC 2.0 错误码

| 错误码 | 含义 | 适用场景 |
|--------|------|---------|
| -32700 | Parse error | JSON 格式不合法 |
| -32600 | Invalid Request | 请求格式不合法 |
| -32601 | Method not found | 调用的方法不存在 |
| -32602 | Invalid params | 参数校验失败 |
| -32603 | Internal error | 网关内部错误 |
| -32001 | Authentication failed | Bearer Token 无效/过期/已吊销 |
| -32002 | Tool not found | 调用的 tool 不存在或用户无权访问 |
| -32003 | Permission denied | VIP 等级不足，无权访问该 tool |
| -32004 | Rate limited | 网关层工具调用频率超限 |
| -32005 | Backend unavailable | 后端不可用/超时/5xx |

> 后端业务错误（HTTP 400/401/403/404/429）不使用 JSON-RPC error，而是通过 `content + isError: true` 返回（工具调用已成功派发，后端返回业务错误）。上表错误码仅用于网关/协议级错误。
