# 套餐管理模块 - API接口

## 1. 文档概述

本文档定义 **套餐管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/packages`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。
> 订单创建/支付/退款/自动续费接口见[订单管理模块 API接口](../../订单管理/API接口.md)。

## 2. 接口清单

### 2.1 用户端接口

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/packages` | GET | 在售套餐列表 | - |
| `/api/v1/packages/{id}` | GET | 套餐详情（下架套餐返回"套餐已下架"） | - |
| `/api/v1/packages/calculate-price` | GET | 价格计算（下单前预览：促销折扣 + 优惠券抵扣） | - |
| `/api/v1/packages/coupons/my` | GET | 我的优惠券列表（按状态筛选） | - |
| `/api/v1/packages/coupons/{id}/receive` | POST | 领取优惠券 | - |

### 2.2 后台管理接口

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/packages/page` | GET | 套餐分页列表 | - |
| `/api/v1/packages` | POST | 新增套餐 | package:add |
| `/api/v1/packages/{id}/form` | GET | 获取套餐表单数据 | - |
| `/api/v1/packages/{id}` | PUT | 修改套餐 | package:edit |
| `/api/v1/packages/{id}/status` | PUT | 上架/下架 | package:edit |
| `/api/v1/packages/{ids}` | DELETE | 删除套餐 | package:delete |
| `/api/v1/packages/sales/stats` | GET | 销售统计 | package:sales |
| `/api/v1/packages/coupons/page` | GET | 优惠券分页列表 | - |
| `/api/v1/packages/coupons` | POST | 创建优惠券 | package:coupon:add |
| `/api/v1/packages/coupons/batch` | POST | 批量发放优惠券 | package:coupon:distribute |
| `/api/v1/packages/coupons/{id}` | PUT | 修改优惠券 | package:coupon:edit |
| `/api/v1/packages/coupons/{ids}` | DELETE | 删除优惠券 | package:coupon:delete |
| `/api/v1/packages/promotions/page` | GET | 促销活动分页列表 | - |
| `/api/v1/packages/promotions` | POST | 创建促销活动 | package:promotion:add |
| `/api/v1/packages/promotions/{id}` | PUT | 修改促销活动 | package:promotion:edit |
| `/api/v1/packages/promotions/{id}/status` | PUT | 促销活动上架/下架 | package:promotion:edit |
| `/api/v1/packages/promotions/{id}` | DELETE | 删除促销活动 | package:promotion:delete |
| `/api/v1/packages/promotions/{id}/packages` | PUT | 关联套餐（维护活动-套餐关联） | package:promotion:edit |

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| package:add | 新增套餐 | 按钮显示 + 接口校验 |
| package:edit | 编辑/上下架套餐 | 按钮显示 + 接口校验 |
| package:delete | 删除套餐 | 按钮显示 + 接口校验 |
| package:sales | 销售统计 | 按钮显示 + 接口校验 |
| package:coupon:add | 优惠券创建 | 按钮显示 + 接口校验 |
| package:coupon:distribute | 优惠券批量发放 | 按钮显示 + 接口校验 |
| package:coupon:edit | 优惠券编辑 | 按钮显示 + 接口校验 |
| package:coupon:delete | 优惠券删除 | 按钮显示 + 接口校验 |
| package:promotion:add | 促销活动创建 | 按钮显示 + 接口校验 |
| package:promotion:edit | 促销活动编辑/上下架/关联套餐 | 按钮显示 + 接口校验 |
| package:promotion:delete | 促销活动删除 | 按钮显示 + 接口校验 |

## 4. 状态枚举

### 4.1 套餐状态

| 状态值 | 说明 |
|--------|------|
| 1 | 在售：用户端可见，可购买 |
| 0 | 下架：用户端不可见，不可购买 |

### 4.2 用户优惠券状态

| 状态值 | 说明 |
|--------|------|
| 1 | 未使用：已领取待使用 |
| 2 | 已使用：已在订单中核销 |
| 3 | 已过期：超过有效期 |
| 4 | 已锁定：已被订单预占，待支付 |

### 4.3 计费周期

| 值 | 说明 |
|----|------|
| monthly | 月卡 |
| quarterly | 季卡 |
| yearly | 年卡 |

> 有效期天数由套餐的 `period_days` 字段控制（月卡 30 / 季卡 90 / 年卡 365 为常见配置，非固定值）。

### 4.4 促销活动状态

| 状态值 | 说明 |
|--------|------|
| 1 | 上架：活动参与价格计算 |
| 0 | 下架：活动不参与价格计算 |

### 4.5 促销活动类型

| 值 | 说明 | 折扣方式 |
|----|------|---------|
| `discount` | 限时折扣 | percent 百分比 / fixed 固定金额 |
| `new_user` | 新用户专享 | percent / fixed（购买时校验新用户） |
| `holiday` | 节日促销 | percent / fixed |
| `full_reduction` | 满减活动 | 满减档位（满 X 减 Y，支持多档位） |

### 4.6 续费叠加最大有效期

续费叠加最大有效期上限 **3 年（1095 天）**，叠加后超过 1095 天的购买请求拒绝。

## 5. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 套餐不存在、优惠券不存在 |
| `A0500` | 业务异常 | 促销价高于原价、计费周期非法、套餐名称已被占用、套餐已下架、套餐存在订单无法删除、套餐参与进行中促销无法下架、优惠券已领完、超过限领数量、优惠券不适用该套餐、优惠券状态无效、新用户专享套餐非新用户购买、续费叠加超过最大有效期（1095 天）、体验券每人限领 1 次 |
| `A0230` | token无效或已过期 | 未登录访问 |

> **说明**：具体错误信息随响应返回（如"促销价不能高于原价""套餐名称已被历史记录占用，无法重复创建"等），错误码统一为 `A0401`/`A0500`。
