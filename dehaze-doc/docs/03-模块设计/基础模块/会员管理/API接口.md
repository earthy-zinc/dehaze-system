# 会员管理模块 - API接口

## 1. 文档概述

本文档定义 **会员管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/members`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/members`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/members/profile` | GET | 当前用户会员信息 | - |
| `/api/v1/members/growth-logs` | GET | 成长值变动明细 | - |
| `/api/v1/members/sign-in` | POST | 每日签到 | - |
| `/api/v1/members/sign-in/calendar` | GET | 签到日历 | - |

> **用户端接口鉴权约定**：上述接口均为用户端接口，`userId` 由服务端从登录态获取（Java `SecurityUtils.getUserId()`、Python `get_current_user_id()`、Go `database.GetUserID(ctx)`），**不接受外部传入**，防止越权查询他人数据。

### 2.2 后台管理接口

**基础路径**：`/api/v1/members`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/members/page` | GET | 会员分页列表 | - |
| `/api/v1/members/{userId}` | GET | 会员详情 | - |
| `/api/v1/members/{userId}/level` | PUT | 等级调整 | member:level:edit |
| `/api/v1/members/{userId}/growth` | PUT | 成长值调整 | member:growth:edit |
| `/api/v1/members/{userId}/status` | PUT | 冻结/解冻 | member:status:edit |
| `/api/v1/members/benefits` | GET | 权益配置列表 | - |
| `/api/v1/members/benefits/{level}` | PUT | 修改权益配置 | member:benefit:edit |

> **导入导出接口**：会员模块的导出（`GET/POST /api/v1/members/_export`）、导入（`POST /api/v1/members/_import`）、模板下载（`GET /api/v1/members/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §8.2 通用CRUD接口模板。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| member:level:edit | 等级调整 | 按钮显示 + 接口校验 |
| member:growth:edit | 成长值调整 | 按钮显示 + 接口校验 |
| member:status:edit | 冻结/解冻 | 按钮显示 + 接口校验 |
| member:benefit:edit | 权益配置 | 按钮显示 + 接口校验 |

## 4. 状态枚举

### 4.1 会员状态

| 状态值 | 显示 | 说明 |
|--------|------|------|
| 1 | 正常（绿色标签） | 会员可正常使用所有功能 |
| 0 | 冻结（红色标签） | 会员无法使用付费功能 |

### 4.2 会员等级

| 等级值 | 名称 | 成长值区间 |
|--------|------|-----------|
| 0 | 普通用户 | 0 - 999 |
| 1 | VIP1 | 1000 - 4999 |
| 2 | VIP2 | 5000 - 19999 |
| 3 | SVIP | ≥20000 |

## 5. 业务错误码

| 错误码 | 错误信息 | 触发场景 |
|-------|---------|---------|
| MEMBER_NOT_FOUND | 会员不存在 | 查询/操作时会员记录不存在 |
| MEMBER_FROZEN | 会员已冻结 | 冻结状态会员尝试使用付费功能 |
| SIGN_IN_ALREADY | 今日已签到 | 重复签到 |
| GROWTH_INSUFFICIENT | 成长值不足 | 扣减成长值时余额不足 |
| LEVEL_ADJUST_FORBIDDEN | 不允许跨级调整 | 等级调整违反逐级升降规则 |
| BENEFIT_CONFIG_INVALID | 权益配置参数无效 | 权益配置缺少必填项或数值非法 |
