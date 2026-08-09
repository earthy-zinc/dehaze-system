# AI 知识库模块 API 接口

## 1. 文档概述

本文档定义 **AI 知识库** 模块的 HTTP API 规范。

- **基础路径**：`/api/v1/kb`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

## 2. 接口清单

### 2.1 知识库管理

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb` | POST | 创建知识库 | `kb:manage` | F-KB-001 |
| `/api/v1/kb` | GET | 知识库列表 | - | F-KB-001 |
| `/api/v1/kb/{id}` | GET | 知识库详情（含配置、统计信息） | - | F-KB-001 |
| `/api/v1/kb/{id}` | PUT | 编辑知识库（名称/描述/检索策略） | `kb:manage` | F-KB-001 |
| `/api/v1/kb/{id}` | DELETE | 删除知识库 | `kb:manage` | F-KB-001 |

### 2.2 文档管理

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb/{id}/documents` | POST | 上传文档（`file_id` 关联已上传的文件） | `kb:document:manage` | F-KB-002 |
| `/api/v1/kb/{id}/documents/batch` | POST | 批量上传文档 | `kb:document:manage` | F-KB-002 |
| `/api/v1/kb/{id}/documents/import-url` | POST | 导入网页为文档 | `kb:document:manage` | F-KB-002 |
| `/api/v1/kb/{id}/documents/text` | POST | 自定义文本创建文档 | `kb:document:manage` | F-KB-002 |
| `/api/v1/kb/{id}/documents` | GET | 知识库文档列表（含处理状态） | - | F-KB-002 |
| `/api/v1/kb/documents/{id}` | GET | 文档详情（含解析后内容） | - | F-KB-002 |
| `/api/v1/kb/documents/{id}` | DELETE | 删除文档及关联分块 | `kb:document:manage` | F-KB-002 |
| `/api/v1/kb/documents/{id}/reprocess` | POST | 重新处理文档 | `kb:document:manage` | F-KB-002 |

### 2.3 分块管理

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb/documents/{id}/chunks` | GET | 文档分块列表 | - | F-KB-003 |
| `/api/v1/kb/documents/{id}/chunks/preview` | GET | 文档分块预览（上传文档时的分块效果预览） | `kb:document:manage` | F-KB-003 |

### 2.4 检索接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb/search` | POST | 知识库检索（支持多知识库、元数据过滤、Rerank） | - | F-KB-004 |
| `/api/v1/kb/{id}/retrieve/test` | POST | 检索测试（知识库管理页面的调试工具） | `kb:manage` | F-KB-004 |

### 2.5 MCP Tool 暴露

知识库检索接口可通过 MCP 网关暴露为 MCP tool，供 AI 对话及第三方 Agent 调用：

| tool 名称 | 对应接口 | 输入参数 | 说明 |
|-----------|---------|---------|------|
| `kb_search` | `POST /api/v1/kb/search` | query、knowledgeBaseIds(可选)、topK(可选)、filters(可选) | 认证复用 API Key |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 查询类接口（列表/详情/分块/检索）无需特殊权限，但按可见性过滤：私有库仅创建者可见，公共库全员只读 |
| `kb:manage` | 知识库管理（创建、编辑、删除、检索测试）。登录用户可管理自有私有库（受 VIP 配额限制）；公共库仅管理员可管理 |
| `kb:document:manage` | 文档管理（上传、删除、重新处理、分块预览），权限规则同 `kb:manage` |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 知识库/文档/分块不存在 |
| `A0500` | 业务异常 | 知识库名称重复、上传文件格式不支持、解析失败、文档处理中不允许删除、私有库数量/单库文档数/分块数超限 |
| `A0230` | token无效或已过期 | 未登录访问 |
| `A0301` | 访问未授权 | 非创建者操作他人私有库、普通用户管理公共库 |
