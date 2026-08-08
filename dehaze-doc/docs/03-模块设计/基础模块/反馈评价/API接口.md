# 反馈评价模块 - API接口

## 1. 文档概述

本文档定义 **反馈评价** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/feedback`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/feedback`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/feedback/ratings` | POST | 提交评分 | - |
| `/api/v1/feedback/ratings/{id}` | PUT | 修改评分 | - |
| `/api/v1/feedback/ratings/my` | GET | 我的评价列表 | - |
| `/api/v1/feedback/ratings/by-prediction/{predictionLogId}` | GET | 按处理记录查评价（仅限本人处理记录） | - |
| `/api/v1/feedback` | POST | 提交反馈 | - |
| `/api/v1/feedback/my` | GET | 我的反馈列表 | - |
| `/api/v1/feedback/{id}` | GET | 反馈详情（非管理员仅限本人反馈） | - |
| `/api/v1/feedback/{id}/supplement` | POST | 补充说明（仅限本人反馈） | - |

### 2.2 后台管理接口

**基础路径**：`/api/v1/feedback`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/feedback/ratings/page` | GET | 评价分页列表 | - |
| `/api/v1/feedback/ratings/{id}/hide` | PUT | 隐藏评价 | feedback:rating:edit |
| `/api/v1/feedback/ratings/{id}/reply` | POST | 回复评价 | feedback:rating:reply |
| `/api/v1/feedback/ratings/stats` | GET | 评价统计 | - |
| `/api/v1/feedback/page` | GET | 反馈分页列表 | - |
| `/api/v1/feedback/{id}/assign` | PUT | 分配处理人 | feedback:assign |
| `/api/v1/feedback/{id}/reply` | POST | 回复反馈 | feedback:reply |
| `/api/v1/feedback/{id}/close` | PUT | 关闭反馈 | feedback:close |
| `/api/v1/feedback/{id}/tags` | PUT | 设置反馈标签 | feedback:edit |
| `/api/v1/feedback/stats` | GET | 反馈统计 | - |

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| feedback:rating:edit | 隐藏评价 | 按钮显示 + 接口校验 |
| feedback:rating:reply | 回复评价 | 按钮显示 + 接口校验 |
| feedback:assign | 分配处理人 | 按钮显示 + 接口校验 |
| feedback:reply | 回复反馈 | 按钮显示 + 接口校验 |
| feedback:close | 关闭反馈 | 按钮显示 + 接口校验 |
| feedback:edit | 反馈标签管理 | 按钮显示 + 接口校验 |

> **说明**：评价分页列表（`/ratings/page`）、评价统计（`/ratings/stats`）、反馈分页列表（`/page`）、反馈统计（`/stats`）接口无需权限标识，登录即可访问。

## 4. 状态枚举

### 4.1 反馈状态

| 状态值 | 说明 |
|--------|------|
| pending | 待处理：用户已提交，等待管理员处理 |
| processing | 处理中：已分配处理人，正在处理 |
| replied | 已回复：管理员已回复 |
| closed | 已关闭：反馈已关闭 |

### 4.2 反馈类型

| 值 | 说明 |
|----|------|
| suggestion | 功能建议 |
| bug | 问题报告 |
| experience | 体验反馈 |
| complaint | 投诉 |

> 评价标签（正面/负面）为业务常量，见 [需求规格.md](./需求规格.md) §2.1.4。

## 5. 业务错误码

| 错误码 | 错误信息 | 触发场景 |
|-------|---------|---------|
| RATING_ALREADY_EXISTS | 该处理记录已评价 | 重复提交评价 |
| RATING_NOT_FOUND | 评价不存在 | 查询/操作时评价不存在，或越权修改他人评价 |
| RATING_EXPIRED | 已超过评价时限 | 处理完成超过 30 天后提交评价（仅创建时校验） |
| FEEDBACK_NOT_FOUND | 反馈不存在 | 查询/操作时反馈不存在（含非管理员越权访问他人反馈） |
| FEEDBACK_CLOSED | 反馈已关闭 | 对已关闭的反馈进行操作 |
| FEEDBACK_LIMIT_EXCEEDED | 今日反馈次数已达上限 | 每用户每日超过 5 条反馈 |
| PREDICTION_LOG_NOT_FOUND | 处理记录不存在 | 评价关联的处理记录不存在 |
| OPERATION_NOT_ALLOW | 操作不允许 | 评价创建时处理记录未完成或不属于当前用户；越权查询他人处理记录的评价 |
