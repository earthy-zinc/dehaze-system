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

| levelCode | 等级值 | 名称 | 成长值区间 |
|-----------|--------|------|-----------|
| `level_0` | 0 | 普通用户 | 0 - 999 |
| `level_1` | 1 | VIP1 | 1000 - 4999 |
| `level_2` | 2 | VIP2 | 5000 - 19999 |
| `level_3` | 3 | SVIP | ≥20000 |

### 4.3 等级来源 (level_source)

区分会员等级的获得方式，控制自动升降级行为：

| 值 | 说明 | 自动降级 |
|----|------|---------|
| `growth` | 成长值达标 | 是，成长值低于等级下限时自动降级 |
| `purchase` | 套餐购买 | 否，套餐有效期内不因成长值变动降级 |
| `admin` | 管理员调整 | 否，管理员指定的等级不自动降级 |

### 4.4 成长值变动类型 (change_type)

| 值 | 说明 |
|----|------|
| `dehaze` | 去雾处理 |
| `evaluate` | 效果评估 |
| `rating` | 提交评价 |
| `sign_in` | 每日签到 |
| `sign_in_bonus` | 连续签到奖励 |
| `consume` | 购买套餐 |
| `refund_deduct` | 退款扣减 |
| `admin_adjust` | 管理员调整 |

### 4.5 处理优先级 (priority)

权益配置中的处理优先级枚举：

| 值 | 说明 |
|----|------|
| 1 | 普通 |
| 2 | 优先 |
| 3 | 高优先 |
| 4 | 最高 |

## 5. 业务错误码

会员管理模块错误码均映射到全局状态码体系（参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §5），三端（Java/Go/Python）统一使用以下定义：

| 错误码常量 | 全局码 | 错误信息 | 触发场景 |
|-------|-------|---------|---------|
| MEMBER_NOT_FOUND | `A0510` | 会员不存在 | 查询/操作时会员记录不存在 |
| MEMBER_FROZEN | `A0511` | 会员已冻结 | 冻结状态会员尝试使用付费功能 |
| SIGN_IN_ALREADY | `A0512` | 今日已签到 | 重复签到 |
| GROWTH_INSUFFICIENT | `A0513` | 成长值不足 | 扣减成长值时余额不足 |
| BENEFIT_CONFIG_INVALID | `A0514` | 权益配置无效 | 权益配置缺少必填项或数值非法 |
