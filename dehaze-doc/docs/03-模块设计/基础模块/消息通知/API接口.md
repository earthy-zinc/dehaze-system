# 消息通知模块 - API接口

## 1. 文档概述

本文档定义 **消息通知** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/messages`（用户端）、`/api/v1/announcements`（管理端）、`/api/v1/message-templates`（管理端）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/messages`

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/messages` | GET | 消息列表（分页） | - | F-MN-002 |
| `/api/v1/messages/unread-count` | GET | 未读消息数 | - | F-MN-003 |
| `/api/v1/messages/{id}` | GET | 消息详情（仅查询，不触发已读） | - | F-MN-003 |
| `/api/v1/messages/{id}/_read` | PATCH | 标记单条已读 | - | F-MN-003 |
| `/api/v1/messages/_read-all` | PATCH | 全部标记已读 | - | F-MN-003 |
| `/api/v1/messages/{ids}` | DELETE | 删除消息（支持批量） | - | F-MN-003 |
| `/api/v1/messages/search` | GET | 搜索消息 | - | F-MN-003 |

**基础路径**：`/api/v1/notification-settings`

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/notification-settings` | GET | 获取通知偏好设置 | - | F-MN-004 |
| `/api/v1/notification-settings` | PATCH | 更新通知偏好设置 | - | F-MN-004 |

### 2.2 后台管理接口

**基础路径**：`/api/v1/announcements`

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/announcements/page` | GET | 公告分页列表 | - | F-MN-005 |
| `/api/v1/announcements` | POST | 创建公告 | `notify:announcement:add` | F-MN-005 |
| `/api/v1/announcements/{id}` | GET | 公告详情 | - | F-MN-005 |
| `/api/v1/announcements/{id}` | PUT | 编辑公告（仅草稿/待发送） | `notify:announcement:edit` | F-MN-005 |
| `/api/v1/announcements/{id}` | DELETE | 删除公告 | `notify:announcement:delete` | F-MN-005 |
| `/api/v1/announcements/{id}/_send` | POST | 立即发送公告 | `notify:announcement:send` | F-MN-005 |
| `/api/v1/announcements/{id}/_cancel` | PATCH | 取消定时公告 | `notify:announcement:cancel` | F-MN-005 |

**基础路径**：`/api/v1/message-templates`

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/message-templates/page` | GET | 模板分页列表 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | GET | 模板详情 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | PUT | 编辑模板 | `notify:template:edit` | F-MN-006 |

### 2.3 内部发送接口（供业务模块调用）

**基础路径**：`/api/v1/messages`

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/messages/send` | POST | 发送消息（业务模块调用） | - | F-MN-001 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `notify:announcement:add` | 创建公告 |
| `notify:announcement:edit` | 编辑公告 |
| `notify:announcement:delete` | 删除公告 |
| `notify:announcement:send` | 发送公告 |
| `notify:announcement:cancel` | 取消定时公告 |
| `notify:template:edit` | 编辑消息模板 |
| - | 其他接口登录用户即可访问 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0550` | 消息不存在 | 查询/操作不存在的消息 |
| `A0552` | 公告不存在 | 查询/操作不存在的公告 |
| `A0553` | 公告状态不允许此操作 | 编辑已发送公告、取消非待发送公告、发送非草稿/待发送公告 |
| `A0554` | 发送范围为空 | 指定用户发送但用户列表为空 |
| `A0555` | 消息模板不存在 | 引用不存在的模板编码 |
| `A0556` | 模板变量缺失 | 使用模板但未提供全部必填变量 |
| `A0558` | 模板已禁用 | 使用已禁用的模板发送消息 |
