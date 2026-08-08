# 消息通知模块 - API接口

## 1. 文档概述

本文档定义 **消息通知** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/messages`（用户端）、`/api/v1/announcements`（管理端）、`/api/v1/message-templates`（管理端）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/messages`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/messages` | GET | 消息列表（分页） | - | F-MN-002 |
| `/api/v1/messages/unread-count` | GET | 未读消息数 | - | F-MN-003 |
| `/api/v1/messages/{id}` | GET | 消息详情（仅查询，不触发已读） | - | F-MN-003 |
| `/api/v1/messages/{id}/_read` | PATCH | 标记单条已读 | - | F-MN-003 |
| `/api/v1/messages/_read-all` | PATCH | 全部标记已读 | - | F-MN-003 |
| `/api/v1/messages/{ids}` | DELETE | 删除消息（支持批量） | - | F-MN-003 |
| `/api/v1/messages/search` | GET | 搜索消息 | - | F-MN-003 |

**基础路径**：`/api/v1/notification-settings`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/notification-settings` | GET | 获取通知偏好设置 | - | F-MN-004 |
| `/api/v1/notification-settings` | PATCH | 更新通知偏好设置（部分更新，深合并） | - | F-MN-004 |

### 2.2 后台管理接口

**基础路径**：`/api/v1/announcements`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/announcements/page` | GET | 公告分页列表 | - | F-MN-005 |
| `/api/v1/announcements` | POST | 创建公告 | `notify:announcement:add` | F-MN-005 |
| `/api/v1/announcements/{id}` | GET | 公告详情 | - | F-MN-005 |
| `/api/v1/announcements/{id}` | PUT | 编辑公告（仅草稿/待发送） | `notify:announcement:edit` | F-MN-005 |
| `/api/v1/announcements/{id}` | DELETE | 删除公告 | `notify:announcement:delete` | F-MN-005 |
| `/api/v1/announcements/{id}/_send` | POST | 立即发送公告 | `notify:announcement:send` | F-MN-005 |
| `/api/v1/announcements/{id}/_cancel` | PATCH | 取消定时公告 | `notify:announcement:cancel` | F-MN-005 |

**基础路径**：`/api/v1/message-templates`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/message-templates/page` | GET | 模板分页列表 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | GET | 模板详情 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | PUT | 编辑模板 | `notify:template:edit` | F-MN-006 |

### 2.3 内部发送接口（供业务模块调用）

**基础路径**：`/api/v1/messages`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/messages/send` | POST | 发送消息（业务模块调用） | - | F-MN-001 |

> **说明**：发送接口要求登录用户身份，**无独立 API Key/权限标识**（早期设计稿的 `internal:notify:send` 权限标识不存在）；由各业务模块（会员/反馈等）调用触发消息生成。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| `notify:announcement:add` | 创建公告 | 按钮显示 + 接口校验 |
| `notify:announcement:edit` | 编辑公告 | 按钮显示 + 接口校验 |
| `notify:announcement:delete` | 删除公告 | 按钮显示 + 接口校验 |
| `notify:announcement:send` | 发送公告 | 按钮显示 + 接口校验 |
| `notify:announcement:cancel` | 取消定时公告 | 按钮显示 + 接口校验 |
| `notify:template:edit` | 编辑消息模板 | 按钮显示 + 接口校验 |

## 4. 状态枚举

### 4.1 消息类型

| 值 | 说明 |
|----|------|
| `inbox` | 站内信 |
| `announcement` | 系统公告 |
| `business` | 业务通知 |
| `member` | 会员通知 |
| `alert` | 告警通知 |
| `critical_alert` | 严重告警 |

### 4.2 消息优先级

| 值 | 说明 | 推送策略 |
|----|------|---------|
| 1 | 低 | 仅站内信 |
| 2 | 中 | 站内信 + APP推送（可选） |
| 3 | 高 | 站内信 + APP推送 |
| 4 | 紧急 | 全渠道立即推送 |

### 4.3 消息已读状态

| 值 | 说明 |
|----|------|
| 0 | 未读 |
| 1 | 已读 |

### 4.4 公告状态

| 值 | 说明 |
|----|------|
| 1 | 草稿：管理员创建未发送，可编辑/发送/删除 |
| 2 | 待发送：定时发送等待中，可编辑/发送/取消/删除 |
| 3 | 已发送：已推送给目标用户，不可编辑/取消 |
| 4 | 已取消：管理员取消定时发送，不可编辑 |

### 4.5 公告类型

| 值 | 说明 |
|----|------|
| `maintenance` | 系统维护 |
| `feature` | 功能更新 |
| `activity` | 活动通知 |
| `operation` | 运营公告 |

### 4.6 公告发送范围

| 值 | 说明 |
|----|------|
| `all` | 全体用户 |
| `level` | 按会员等级 |
| `specified` | 指定用户 |

## 5. 业务错误码

| 错误码 | 常量名 | 错误信息 | 触发场景 | 状态 |
|-------|--------|---------|---------|------|
| `A0550` | MESSAGE_NOT_FOUND | 消息不存在 | 查询/操作不存在的消息；**操作他人消息时同样返回（不暴露存在性）** | ✅ 使用中 |
| `A0551` | MESSAGE_NO_PERMISSION | 无权操作此消息 | 操作非自己的消息 | ⚠️ 已定义未使用（越权统一返回 MESSAGE_NOT_FOUND） |
| `A0552` | ANNOUNCEMENT_NOT_FOUND | 公告不存在 | 查询/操作不存在的公告 | ✅ 使用中 |
| `A0553` | ANNOUNCEMENT_STATUS_INVALID | 公告状态不允许此操作 | 编辑已发送公告、取消非待发送公告、发送非草稿/待发送公告 | ✅ 使用中 |
| `A0554` | ANNOUNCEMENT_TARGET_EMPTY | 发送范围为空 | 指定用户发送但用户列表为空 | ✅ 使用中 |
| `A0555` | MESSAGE_TEMPLATE_NOT_FOUND | 消息模板不存在 | 引用不存在的模板编码 | ✅ 使用中 |
| `A0556` | TEMPLATE_VAR_MISSING | 模板变量缺失 | 使用模板但未提供全部必填变量 | ✅ 使用中 |
| `A0557` | NOTIFICATION_SETTING_NOT_FOUND | 通知设置不存在 | 用户未初始化通知设置 | ⚠️ 已定义未使用（获取设置时 upsert 自动初始化） |
| `A0558` | TEMPLATE_DISABLED | 模板已禁用 | 使用已禁用的模板发送消息 | ✅ 使用中 |
| `A0559` | MESSAGE_ALREADY_READ | 消息已读 | 重复标记已读 | ⚠️ 已定义未使用（标记已读幂等静默成功） |
