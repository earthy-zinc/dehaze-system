# AI 模型管理模块 API 接口

## 1. 文档概述

本文档定义 **AI 模型管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。本模块承载平台 AI 模型（chat/embedding/rerank）与模型供应商（含 API Key）的统一管理。

- **基础路径**：`/api/v1/ai`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **消费方**：AI 对话（会话内 chat 模型选择）、AI 知识库（按 `model_type` 选择 embedding/rerank 模型）

## 2. 接口清单

### 2.1 模型供应商管理接口（AiProviderAPI）

> 供应商与 API Key 接口均需 `ai:model:manage` 权限，`/providers/enabled` 除外。供应商不限于对话 LLM，覆盖 chat/embedding/rerank 全部模型类型的供应商接入。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/providers` | GET | 供应商分页列表（含健康状态：健康/可疑/熔断） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/enabled` | GET | 启用供应商列表（普通用户只读，不含敏感信息） | - | F-M08-007 |
| `/api/v1/ai/providers` | POST | 新增供应商（保存后异步触发连通性测试，结果仅提示不阻断） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}` | PUT | 更新供应商 | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}` | DELETE | 删除供应商（逻辑删除） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/test-connection` | POST | 连通性测试（不产生计费） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/circuit/close` | POST | 手动解除供应商熔断状态 | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys` | GET | 供应商下的 API Key 列表（不返回明文） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys` | POST | 新增 API Key（请求体携带明文，服务端加密存储） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys/{keyId}` | PUT | 更新 API Key（优先级、权重、状态、限额） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/providers/{id}/keys/{keyId}` | DELETE | 删除 API Key（物理删除，校验至少保留一个启用 Key） | `ai:model:manage` | F-M08-007 |

### 2.2 模型管理接口（AiModelAPI）

> 模型管理分双端：管理端分页列表需 `ai:model:manage` 权限；`/models/enabled` 无需特殊权限，按登录用户 VIP 等级过滤。两者均支持 `modelType` 查询参数按模型类型筛选。更新/删除以 **`model_id` 字符串**为路径参数（非自增主键）。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/models` | GET | 模型分页列表（管理端，支持 `modelType` 筛选） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/enabled` | GET | 启用模型列表（用户端/消费方，按 VIP 过滤，支持 `modelType` 筛选） | - | F-M08-007 |
| `/api/v1/ai/models` | POST | 新增模型配置 | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}` | PUT | 更新模型配置（`model_type` / `dimension` 创建后不可修改） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}` | DELETE | 删除模型配置（逻辑删除，`model_id` 不可复用） | `ai:model:manage` | F-M08-007 |

### 2.3 模型用户售价接口（AiModelAPI）

> 用户售价按版本管理：同模型同供应商新增即生成新版本（版本号递增），历史版本保留可追溯。每个版本含三维档位明细（token 类型 `input`/`cached`/`output` × 时段 `peak`/`idle` × 上下文分段 `min_tokens`~`max_tokens`）。分页参数为 `page`/`size`，与管理端分页的 `pageNum`/`pageSize` 不同。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/models/{model_id}/prices` | GET | 价格版本分页列表（含档位明细） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}/prices` | POST | 新增价格版本（生成新版本号，含档位明细） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}/prices/{price_id}` | PUT | 更新价格版本主表字段（单价单位/生效时间/状态） | `ai:model:manage` | F-M08-007 |
| `/api/v1/ai/models/{model_id}/prices/{price_id}` | DELETE | 删除价格版本（主表与档位明细一并逻辑删除） | `ai:model:manage` | F-M08-007 |

> 档位单价 `unit_price` 后端为 `Decimal`，响应序列化为**字符串**，前端参与计算前需转 Number。更新接口只改主表字段，档位明细不支持局部更新（调价走新增版本）。

### 2.4 运营统计接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai/usage/stats` | GET | 用量与成本统计（供应商健康看板、模型用量分布、降级与故障统计），支持时间范围（start/end）与粒度（day/hour）聚合 | `ai:model:manage` | F-M08-007 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `ai:model:manage` | 模型与供应商配置管理（模型、供应商、API Key 的新增/修改/删除及运营统计），仅管理员；越权返回 403（`A0301`） |
| - | 启用模型/供应商列表查询，登录用户可用；消费方（AI 对话、AI 知识库）按模型类型过滤使用 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0230` | token 无效或已过期 | 未登录访问 |
| `A0301` | 访问未授权 | 无 `ai:model:manage` 权限调用管理端接口 |
| `A0401` | 请求资源不存在 | 模型/供应商/API Key 不存在 |
| `A0501` | 数据已存在 | `model_id` / `provider_code` / API Key 明文查重命中（含逻辑删除历史，不可复用） |
| `A0504` | 存在关联数据，无法删除 | 删除被活跃会话引用、被知识库引用或被关联模型引用的模型/供应商 |
