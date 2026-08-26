# AI 模型管理模块 API 接口

## 1. 文档概述

本文档定义 **AI 模型管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。本模块承载平台 AI 模型（chat/embedding/rerank）与模型供应商（含 API Key）的统一管理。

- **基础路径**：`/api/v1/ai`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **消费方**：AI 对话（会话内模型选择，接口见本模块 + AI对话/API接口.md）、AI 知识库（embedding/rerank 模型选择，创建知识库时按 `model_type` 过滤）

> 本模块原为 AI 对话模块的功能域（F-M08-007），供应商与模型接口随模块抽离于此；AI 对话模块的会话/消息等接口仍在 [AI对话/API接口.md](../../核心模块/AI对话/API接口.md)。

## 2. 接口清单

### 2.1 模型供应商管理接口（AiProviderAPI）

> 供应商与 API Key 相关为管理员接口，需 `ai:model:manage` 权限（后端 `require_permission` 拦截，越权返回 403）；`/providers/enabled` 列表无需特殊权限。**供应商列表为分页结构**（`PageResult`）。供应商不限于对话 LLM，覆盖 chat/embedding/rerank 全部模型类型的供应商接入。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/providers` | GET | 供应商分页列表（管理员，含健康状态：健康/可疑/熔断） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/enabled` | GET | 启用供应商列表（普通用户只读，不含敏感信息） | - | F-M08-007 |
| `/api/v1/ai/providers` | POST | 新增供应商（保存后异步触发连通性测试，结果仅提示不阻断） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}` | PUT | 更新供应商 | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}` | DELETE | 删除供应商（软删除） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/test-connection` | POST | 连通性测试（发送最小探测请求验证地址与凭据，不产生计费） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/circuit/close` | POST | 手动解除供应商熔断状态 | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys` | GET | 供应商下的 API Key 列表（仅返回前缀、状态、优先级、权重、限额、最近使用时间，不返回明文） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys` | POST | 新增 API Key（请求体携带明文，服务端加密存储） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys/{keyId}` | PUT | 更新 API Key（优先级、权重、状态、日额度/频率限额） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys/{keyId}` | DELETE | 删除 API Key（物理删除，校验至少保留一个启用 Key） | `ai:model:manage` | F-M08-007 |

### 2.2 模型管理接口

> 模型管理分双端：**管理端分页列表** `/api/v1/ai/models`（需 `ai:model:manage` 权限）+ **启用模型列表** `/api/v1/ai/models/enabled`（无需特殊权限，按 VIP 过滤；**支持 `modelType` 查询参数过滤**，知识库创建向导按 `embedding` 类型拉取）。update/delete 以 **`model_id` 字符串** 为路径参数（非自增主键）。模型表单含 `modelType`（chat/embedding/rerank）；embedding 模型含 `dimension`（向量维度，创建后不可修改）；chat 模型能力字段平铺（`supportsMultimodal`/`supportsToolCall`/`supportsStreaming`/`supportsPromptCache`/`supportsStructuredOutput`）。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/models` | GET | 模型分页列表（管理端；含模型类型、能力标识、积分单价、上下文长度、速度档位、模型标签、降级标识；支持按 `modelType` 查询参数筛选） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/enabled` | GET | 启用模型列表（用户端/消费方，按 VIP 过滤，支持 `modelType` 查询参数过滤） | - | F-M08-007 |
| `/api/v1/ai/models` | POST | 新增模型配置（管理员；含 `model_type`、embedding `dimension`） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}` | PUT | 更新模型配置（管理员；`model_id` 字符串为路径参数；embedding `dimension` 不可修改） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}` | DELETE | 删除模型配置（管理员，软删除，model_id 不可复用） | `ai:model:manage` | F-M08-007 |

### 2.3 运营统计接口

> 用量与成本分析（管理端运营视图，需 `ai:model:manage`）：供应商健康看板、模型用量分布、降级与故障统计。数据同源调用日志（`sys_ai_billing` 等），支持时间范围（start/end）与粒度（day/hour）聚合。接口定义见 [需求规格.md §2.8.6](./需求规格.md)。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/usage/stats` | GET | 用量与成本统计（供应商健康看板：成功率/429/P95/熔断；模型用量分布：Token/调用数/积分开销；降级与故障统计：降级频率/Key 失败切换），按时间范围与粒度聚合 | `ai:model:manage` | F-M08-007 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 启用模型/供应商列表查询，登录用户可用；消费方（AI 对话、AI 知识库）按模型类型过滤使用 |
| `ai:model:manage` | 模型与供应商配置管理（新增/修改/删除模型、供应商、API Key），仅管理员；越权返回 403（`A0301`） |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 模型/供应商/API Key 不存在 |
| `A0301` | 访问未授权 | 无 `ai:model:manage` 权限调用管理端接口（模型/供应商/Key 管理） |
| `A0501` | 数据已存在 | model_id / provider_code / API Key 明文查重命中（含软删历史，不可复用） |
| `A0504` | 存在关联数据，无法删除 | 删除被活跃会话引用/被知识库引用/被关联供应商引用的模型或供应商 |
| `A0230` | token 无效或已过期 | 未登录访问 |
