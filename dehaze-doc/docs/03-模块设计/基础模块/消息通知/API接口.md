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
| `/api/v1/messages/{id}` | GET | 消息详情 | - | F-MN-003 |
| `/api/v1/messages/{id}/read` | PUT | 标记单条已读 | - | F-MN-003 |
| `/api/v1/messages/read-all` | PUT | 全部标记已读 | - | F-MN-003 |
| `/api/v1/messages/{ids}` | DELETE | 删除消息（支持批量） | - | F-MN-003 |
| `/api/v1/messages/search` | GET | 搜索消息 | - | F-MN-003 |

**基础路径**：`/api/v1/notification-settings`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/notification-settings` | GET | 获取通知偏好设置 | - | F-MN-004 |
| `/api/v1/notification-settings` | PUT | 更新通知偏好设置 | - | F-MN-004 |

### 2.2 后台管理接口

**基础路径**：`/api/v1/announcements`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/announcements/page` | GET | 公告分页列表 | - | F-MN-005 |
| `/api/v1/announcements` | POST | 创建公告 | `notify:announcement:add` | F-MN-005 |
| `/api/v1/announcements/{id}` | GET | 公告详情 | - | F-MN-005 |
| `/api/v1/announcements/{id}` | PUT | 编辑公告（仅草稿/待发送） | `notify:announcement:edit` | F-MN-005 |
| `/api/v1/announcements/{id}` | DELETE | 删除公告 | `notify:announcement:delete` | F-MN-005 |
| `/api/v1/announcements/{id}/send` | POST | 立即发送公告 | `notify:announcement:send` | F-MN-005 |
| `/api/v1/announcements/{id}/cancel` | PUT | 取消定时公告 | `notify:announcement:cancel` | F-MN-005 |

**基础路径**：`/api/v1/message-templates`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/message-templates/page` | GET | 模板分页列表 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | GET | 模板详情 | - | F-MN-006 |
| `/api/v1/message-templates/{id}` | PUT | 编辑模板 | `notify:template:edit` | F-MN-006 |

### 2.3 内部接口（供其他后端服务调用）

**基础路径**：`/api/v1/messages`

| 接口路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|---------|------|---------|---------|-----------|
| `/api/v1/messages/send` | POST | 发送消息（内部调用） | `internal:notify:send` | F-MN-001 |

> 内部接口供 dehaze-python / dehaze-go 等后端服务通过 API Key 鉴权调用，用于业务事件触发消息推送。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| `notify:announcement:add` | 创建公告 | 按钮显示 + 接口校验 |
| `notify:announcement:edit` | 编辑公告 | 按钮显示 + 接口校验 |
| `notify:announcement:delete` | 删除公告 | 按钮显示 + 接口校验 |
| `notify:announcement:send` | 发送公告 | 按钮显示 + 接口校验 |
| `notify:announcement:cancel` | 取消定时公告 | 按钮显示 + 接口校验 |
| `notify:template:edit` | 编辑消息模板 | 按钮显示 + 接口校验 |
| `internal:notify:send` | 内部消息发送 | API Key 鉴权 |

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

| 值 | 显示 | 说明 |
|----|------|------|
| 1 | 草稿（灰色标签） | 管理员创建未发送，可编辑 |
| 2 | 待发送（橙色标签） | 定时发送等待中，可取消 |
| 3 | 已发送（绿色标签） | 已推送给目标用户 |
| 4 | 已取消（灰色标签） | 管理员取消定时发送 |

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
| `tag` | 按用户标签 |
| `specified` | 指定用户 |

## 5. 接口详情

### 5.1 消息列表

`GET /api/v1/messages`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pageNum | int | 否 | 页码，默认 1 |
| pageSize | int | 否 | 每页条数，默认 20 |
| type | string | 否 | 按消息类型筛选：`inbox`/`announcement`/`business`/`member`/`alert`/`critical_alert` |
| readStatus | int | 否 | 按已读状态筛选：`0`(未读)/`1`(已读) |

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "list": [
      {
        "id": 1,
        "type": "member",
        "typeLabel": "会员通知",
        "title": "恭喜您升级至 VIP2",
        "summary": "恭喜您升级至 VIP2，已解锁高清图导出、对比报告导出等权益...",
        "priority": 2,
        "readStatus": 0,
        "senderType": 1,
        "jumpUrl": "/member/profile",
        "createTime": "2026-07-15 14:30:25"
      }
    ],
    "total": 58,
    "pageNum": 1,
    "pageSize": 20
  }
}
```

### 5.2 未读消息数

`GET /api/v1/messages/unread-count`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "count": 12
  }
}
```

### 5.3 消息详情

`GET /api/v1/messages/{id}`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "id": 1,
    "type": "member",
    "typeLabel": "会员通知",
    "title": "恭喜您升级至 VIP2",
    "content": "亲爱的用户，恭喜您成功升级至 VIP2！\n您已解锁以下新权益：\n- 高清图导出\n- 对比报告导出\n- 批量打包下载",
    "priority": 2,
    "senderType": 1,
    "senderTypeLabel": "系统",
    "readStatus": 1,
    "readTime": "2026-07-15 14:35:00",
    "jumpUrl": "/member/profile",
    "extra": null,
    "createTime": "2026-07-15 14:30:25"
  }
}
```

### 5.4 标记单条已读

`PUT /api/v1/messages/{id}/read`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": null
}
```

### 5.5 全部标记已读

`PUT /api/v1/messages/read-all`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| type | string | 否 | 按消息类型标记已读，不传则标记全部 |

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "affectedCount": 12
  }
}
```

### 5.6 删除消息

`DELETE /api/v1/messages/{ids}`

**路径参数**

| 参数 | 说明 |
|------|------|
| ids | 消息ID，多个用逗号分隔，如 `1,2,3` |

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": null
}
```

### 5.7 搜索消息

`GET /api/v1/messages/search`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| keyword | string | 是 | 搜索关键字（匹配标题和正文） |
| pageNum | int | 否 | 页码，默认 1 |
| pageSize | int | 否 | 每页条数，默认 20 |

**响应数据**

同 5.1 消息列表响应结构。

### 5.8 获取通知偏好设置

`GET /api/v1/notification-settings`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "pushEnabled": true,
    "dndEnabled": false,
    "dndStart": "22:00:00",
    "dndEnd": "08:00:00",
    "preferences": {
      "typeChannels": {
        "announcement": { "push": true },
        "business": { "push": false },
        "member": { "push": true }
      },
      "moduleSwitches": {
        "prediction": true,
        "feedback": true,
        "announcement": true
      }
    }
  }
}
```

### 5.9 更新通知偏好设置

`PUT /api/v1/notification-settings`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pushEnabled | boolean | 否 | APP推送总开关 |
| dndEnabled | boolean | 否 | 免打扰开关 |
| dndStart | string | 否 | 免打扰开始时间，格式 `HH:mm:ss` |
| dndEnd | string | 否 | 免打扰结束时间，格式 `HH:mm:ss` |
| preferences | object | 否 | 细粒度偏好设置 |

**请求示例**

```json
{
  "pushEnabled": true,
  "dndEnabled": true,
  "dndStart": "23:00:00",
  "dndEnd": "07:00:00",
  "preferences": {
    "typeChannels": {
      "announcement": { "push": true },
      "business": { "push": false },
      "member": { "push": true }
    },
    "moduleSwitches": {
      "prediction": true,
      "feedback": true,
      "announcement": true
    }
  }
}
```

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": null
}
```

### 5.10 公告分页列表

`GET /api/v1/announcements/page`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pageNum | int | 否 | 页码，默认 1 |
| pageSize | int | 否 | 每页条数，默认 10 |
| title | string | 否 | 公告标题模糊搜索 |
| type | string | 否 | 公告类型筛选 |
| status | int | 否 | 公告状态筛选 |

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "list": [
      {
        "id": 1,
        "title": "系统升级维护通知",
        "type": "maintenance",
        "typeLabel": "系统维护",
        "importance": 2,
        "targetScope": "all",
        "targetScopeLabel": "全体用户",
        "status": 3,
        "statusLabel": "已发送",
        "sendTime": "2026-07-20 10:00:00",
        "expireTime": "2026-07-27 00:00:00",
        "sentCount": 1523,
        "createTime": "2026-07-19 15:30:00",
        "createBy": 1
      }
    ],
    "total": 8,
    "pageNum": 1,
    "pageSize": 10
  }
}
```

### 5.11 创建公告

`POST /api/v1/announcements`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| title | string | 是 | 公告标题，2-50 字符 |
| content | string | 是 | 公告内容 |
| type | string | 是 | 公告类型：`maintenance`/`feature`/`activity`/`operation` |
| importance | int | 是 | 重要级别：`1`(普通)/`2`(重要) |
| targetScope | string | 是 | 发送范围：`all`/`level`/`tag`/`specified` |
| targetParams | object | 否 | 范围参数，targetScope 为 `level` 时传 `{"level": 2}`，为 `specified` 时传 `{"userIds": [1,2,3]}` |
| sendTime | datetime | 否 | 定时发送时间，不传则保存为草稿 |
| expireTime | datetime | 否 | 过期时间 |

**请求示例**

```json
{
  "title": "暑期VIP特惠活动",
  "content": "暑期 VIP 特惠活动开启，年费套餐限时 7 折！活动时间：8月1日-8月31日。",
  "type": "activity",
  "importance": 2,
  "targetScope": "all",
  "sendTime": "2026-08-01 09:00:00",
  "expireTime": "2026-09-01 00:00:00"
}
```

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "id": 2
  }
}
```

### 5.12 公告详情

`GET /api/v1/announcements/{id}`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "id": 1,
    "title": "系统升级维护通知",
    "content": "系统将于 7月30日 02:00-04:00 进行升级维护，届时服务暂不可用。",
    "type": "maintenance",
    "typeLabel": "系统维护",
    "importance": 2,
    "importanceLabel": "重要",
    "targetScope": "all",
    "targetScopeLabel": "全体用户",
    "targetParams": null,
    "status": 3,
    "statusLabel": "已发送",
    "sendTime": "2026-07-20 10:00:00",
    "expireTime": "2026-07-27 00:00:00",
    "sentCount": 1523,
    "createTime": "2026-07-19 15:30:00",
    "updateTime": "2026-07-20 10:00:05"
  }
}
```

### 5.13 编辑公告

`PUT /api/v1/announcements/{id}`

> 仅草稿（status=1）和待发送（status=2）状态的公告可编辑。

**请求参数**

同 5.11 创建公告，所有字段均为可选。

### 5.14 发送公告

`POST /api/v1/announcements/{id}/send`

> 将草稿或待发送状态的公告立即发送给目标用户。发送后状态变为已发送。

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "sentCount": 1523
  }
}
```

### 5.15 取消定时公告

`PUT /api/v1/announcements/{id}/cancel`

> 仅待发送（status=2）状态的公告可取消。取消后状态变为已取消。

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": null
}
```

### 5.16 消息模板分页列表

`GET /api/v1/message-templates/page`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pageNum | int | 否 | 页码，默认 1 |
| pageSize | int | 否 | 每页条数，默认 20 |
| name | string | 否 | 模板名称模糊搜索 |
| type | string | 否 | 消息类型筛选 |
| status | int | 否 | 状态筛选：`1`(启用)/`0`(禁用) |

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "list": [
      {
        "id": 1,
        "code": "member_level_up",
        "name": "会员升级通知",
        "type": "member",
        "titleTemplate": "恭喜您升级至 {levelName}",
        "priority": 2,
        "status": 1,
        "createTime": "2026-07-01 00:00:00"
      }
    ],
    "total": 15,
    "pageNum": 1,
    "pageSize": 20
  }
}
```

### 5.17 消息模板详情

`GET /api/v1/message-templates/{id}`

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "id": 1,
    "code": "member_level_up",
    "name": "会员升级通知",
    "type": "member",
    "titleTemplate": "恭喜您升级至 {levelName}",
    "contentTemplate": "亲爱的用户，恭喜您成功升级至 {levelName}！\n您已解锁以下新权益：\n{benefitList}",
    "priority": 2,
    "channels": {
      "inbox": true,
      "push": true,
      "email": false
    },
    "variables": [
      { "name": "levelName", "desc": "等级名称" },
      { "name": "benefitList", "desc": "权益列表" }
    ],
    "status": 1,
    "createTime": "2026-07-01 00:00:00",
    "updateTime": "2026-07-01 00:00:00"
  }
}
```

### 5.18 编辑消息模板

`PUT /api/v1/message-templates/{id}`

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | 否 | 模板名称 |
| titleTemplate | string | 否 | 标题模板 |
| contentTemplate | string | 否 | 正文模板 |
| priority | int | 否 | 默认优先级 |
| channels | object | 否 | 默认推送渠道 |
| status | int | 否 | 状态 |

### 5.19 内部消息发送

`POST /api/v1/messages/send`

> 供其他后端服务（Python/Go）通过 API Key 鉴权调用，用于业务事件触发消息推送。

**请求头**

| 请求头 | 说明 |
|--------|------|
| `Authorization` | `Bearer dhak_xxx`（API Key） |

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| templateCode | string | 否 | 模板编码（使用模板时传入，自动填充标题和正文） |
| type | string | 是 | 消息类型 |
| title | string | 否 | 消息标题（未使用模板时必填） |
| content | string | 否 | 消息正文（未使用模板时必填） |
| recipientIds | array | 是 | 接收人ID列表 |
| bizModule | string | 否 | 业务模块 |
| bizId | string | 否 | 业务ID（用于幂等去重） |
| priority | int | 否 | 优先级，默认 2 |
| jumpUrl | string | 否 | 跳转链接 |
| variables | object | 否 | 模板变量（使用模板时传入，如 `{"levelName": "VIP2"}`） |
| extra | object | 否 | 扩展数据 |

**请求示例**

```json
{
  "templateCode": "member_level_up",
  "type": "member",
  "recipientIds": [1001],
  "bizModule": "member",
  "bizId": "member_1001_levelup",
  "priority": 3,
  "jumpUrl": "/member/profile",
  "variables": {
    "levelName": "VIP2",
    "benefitList": "- 高清图导出\n- 对比报告导出\n- 批量打包下载"
  }
}
```

**响应数据**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "messageIds": [1]
  }
}
```

## 6. 业务错误码

| 错误码 | 常量名 | 错误信息 | 触发场景 |
|-------|--------|---------|---------|
| `A0550` | MESSAGE_NOT_FOUND | 消息不存在 | 查询/操作不存在的消息 |
| `A0551` | MESSAGE_NO_PERMISSION | 无权操作此消息 | 操作非自己的消息 |
| `A0552` | ANNOUNCEMENT_NOT_FOUND | 公告不存在 | 查询/操作不存在的公告 |
| `A0553` | ANNOUNCEMENT_STATUS_INVALID | 公告状态不允许此操作 | 编辑已发送公告、取消非待发送公告 |
| `A0554` | ANNOUNCEMENT_TARGET_EMPTY | 发送范围为空 | 指定用户发送但用户列表为空 |
| `A0555` | MESSAGE_TEMPLATE_NOT_FOUND | 消息模板不存在 | 引用不存在的模板编码 |
| `A0556` | TEMPLATE_VAR_MISSING | 模板变量缺失 | 使用模板但未提供全部必填变量 |
| `A0557` | NOTIFICATION_SETTING_NOT_FOUND | 通知设置不存在 | 用户未初始化通知设置 |
| `A0558` | TEMPLATE_DISABLED | 模板已禁用 | 使用已禁用的模板发送消息 |
| `A0559` | MESSAGE_ALREADY_READ | 消息已读 | 重复标记已读 |
