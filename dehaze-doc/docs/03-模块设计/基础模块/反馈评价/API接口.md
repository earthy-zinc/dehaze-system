# 反馈评价模块 - API接口

## 1. 文档概述

本文档定义 **反馈评价** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/feedback`

## 2. 接口清单

### 2.1 用户端接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/feedback/ratings` | POST | 提交评分 | - | F-FE-001 |
| `/api/v1/feedback/ratings/{id}` | PUT | 修改评分 | - | F-FE-001 |
| `/api/v1/feedback/ratings/my` | GET | 我的评价列表 | - | F-FE-002 |
| `/api/v1/feedback/ratings/by-prediction/{predictionLogId}` | GET | 按处理记录查评价（仅限本人处理记录） | - | F-FE-002 |
| `/api/v1/feedback` | POST | 提交反馈 | - | F-FE-003 |
| `/api/v1/feedback/my` | GET | 我的反馈列表 | - | F-FE-004 |
| `/api/v1/feedback/{id}` | GET | 反馈详情（非管理员仅限本人反馈） | - | F-FE-004 |
| `/api/v1/feedback/{id}/supplement` | POST | 补充说明（仅限本人反馈） | - | F-FE-004 |

### 2.2 后台管理接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/feedback/ratings/page` | GET | 评价分页列表 | - | F-FE-005 |
| `/api/v1/feedback/ratings/{id}/hide` | PUT | 隐藏评价 | `feedback:rating:edit` | F-FE-005 |
| `/api/v1/feedback/ratings/{id}/reply` | POST | 回复评价 | `feedback:rating:reply` | F-FE-005 |
| `/api/v1/feedback/ratings/stats` | GET | 评价统计 | - | F-FE-008 |
| `/api/v1/feedback/page` | GET | 反馈分页列表 | - | F-FE-006 |
| `/api/v1/feedback/{id}/assign` | PUT | 分配处理人 | `feedback:assign` | F-FE-006 |
| `/api/v1/feedback/{id}/reply` | POST | 回复反馈 | `feedback:reply` | F-FE-006 |
| `/api/v1/feedback/{id}/close` | PUT | 关闭反馈 | `feedback:close` | F-FE-006 |
| `/api/v1/feedback/{id}/tags` | PUT | 设置反馈标签 | `feedback:edit` | F-FE-006 |
| `/api/v1/feedback/stats` | GET | 反馈统计 | - | F-FE-008 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `feedback:rating:edit` | 隐藏评价 |
| `feedback:rating:reply` | 回复评价 |
| `feedback:assign` | 分配处理人 |
| `feedback:reply` | 回复反馈 |
| `feedback:close` | 关闭反馈 |
| `feedback:edit` | 反馈标签管理 |
| - | 评价/反馈查询与统计接口登录用户即可访问 |

> **业务规则**：提交反馈时 `feedbackType` 枚举为 `suggestion/bug/experience/complaint`；`title` 长度 5-50、`content` 长度 10-1000；反馈类型不合规返回 `A0400`。

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0400` | 用户请求参数错误 | 反馈类型非法、标题/内容长度不符、图片数量超限 |
| `A0540` | 该处理记录已评价 | 重复提交评价 |
| `A0541` | 评价不存在 | 查询/操作时评价不存在，或越权修改他人评价 |
| `A0542` | 已超过评价时限 | 处理完成超过 30 天后提交评价 |
| `A0543` | 反馈不存在 | 查询/操作时反馈不存在（含非管理员越权访问他人反馈） |
| `A0544` | 反馈已关闭 | 对已关闭的反馈进行操作 |
| `A0545` | 今日反馈次数已达上限 | 每用户每日超过 5 条反馈 |
| `A0546` | 处理记录不存在 | 评价关联的处理记录不存在 |
