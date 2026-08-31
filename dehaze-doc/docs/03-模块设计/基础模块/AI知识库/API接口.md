# AI 知识库模块 API 接口

## 1. 文档概述

本文档定义 **AI 知识库** 模块的 HTTP API 规范，基础路径 `/api/v1/kb`。

## 2. 接口清单

### 2.1 知识库管理

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb` | POST | 创建知识库（`embedding_model` 从 [AI模型管理模块](../AI模型管理/API接口.md) 注册表选择，`model_type=embedding` 且启用） | `kb:manage` | F-KB-001 |
| `/api/v1/kb` | GET | 知识库列表（默认按可见性过滤；`view=admin` 管理端视角返回全部知识库含私有库只读监控，需 `kb:manage`） | - | F-KB-001 |
| `/api/v1/kb/{id}` | GET | 知识库详情（含配置、统计信息） | - | F-KB-001 |
| `/api/v1/kb/{id}` | PUT | 编辑知识库（名称/描述/检索策略） | `kb:manage` | F-KB-001 |
| `/api/v1/kb/{id}` | DELETE | 删除知识库 | `kb:manage` | F-KB-001 |
| `/api/v1/kb/{id}/index-stats` | GET | 知识库索引状态（ES 索引大小/索引文档数/阈值告警状态，管理端索引状态区） | `kb:audit` | F-KB-001/003 |

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
| `/api/v1/kb/documents/chunks/preview` | POST | 分块预览（基于 `file_id` + 分块配置，返回分块效果预览，不向量化不写索引，供上传前确认） | `kb:document:manage` | F-KB-003 |

### 2.4 检索接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/kb/search` | POST | 知识库检索（支持多知识库、元数据过滤、Rerank） | - | F-KB-004 |
| `/api/v1/kb/{id}/retrieve/test` | POST | 检索测试（知识库管理页面的调试工具） | `kb:manage` | F-KB-004 |
| `/api/v1/kb/{id}/retrieve/test-sets` | POST | 创建召回测试集（问题 + 期望命中段落） | `kb:audit` | F-KB-004 |
| `/api/v1/kb/{id}/retrieve/test-sets` | GET | 召回测试集列表 | `kb:audit` | F-KB-004 |
| `/api/v1/kb/{id}/retrieve/test-sets/{testSetId}/run` | POST | 执行召回测试集（返回 Recall@K 与命中率） | `kb:audit` | F-KB-004 |
| `/api/v1/kb/{id}/chunks/low-quality` | GET | 低质量片段列表（被点踩片段，用于反馈闭环） | `kb:audit` | F-KB-004 |

### 2.5 MCP Tool 暴露

知识库检索接口可通过 MCP 网关暴露为 MCP tool，供 AI 对话及第三方 Agent 调用：

| tool 名称 | 对应接口 | 输入参数 | 说明 |
|-----------|---------|---------|------|
| `kb_search` | `POST /api/v1/kb/search` | query、knowledgeBaseIds(可选)、topK(可选)、filters(可选) | 认证复用 API Key |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 查询类接口（列表/详情/分块/检索）无需特殊权限，但按可见性过滤：私有库仅创建者可见，公共库全员只读 |
| `kb:manage` | 知识库管理（创建、编辑、删除、检索测试）。登录用户可管理自有私有库（受 VIP 配额限制）；公共库仅管理员可管理；`view=admin` 管理端列表接口仅管理员可用 |
| `kb:audit` | 管理端审计（索引状态、召回测试集管理/执行、低质量片段列表）。普通用户持有 `kb:manage` 也无法访问，需 `kb:audit` 权限（ROOT 角色放行） |
| `kb:document:manage` | 文档管理（上传、删除、重新处理、分块预览），权限规则同 `kb:manage` |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 知识库/文档/分块不存在 |
| `A0500` | 业务异常 | 知识库名称重复、上传文件格式不支持、解析失败、文档处理中不允许删除、私有库数量/单库文档数/分块数超限 |
| `A0230` | token无效或已过期 | 未登录访问 |
| `A0301` | 访问未授权 | 非创建者操作他人私有库、普通用户管理公共库 |
