# 套餐管理模块 - API接口

## 1. 文档概述

本文档定义 **套餐管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/packages`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/packages` | GET | 在售套餐列表 | - | F-PM-005 |
| `/api/v1/packages/{id}` | GET | 套餐详情（下架套餐返回"套餐已下架"） | - | F-PM-005 |
| `/api/v1/packages/calculate-price` | GET | 价格计算（下单前预览：促销折扣 + 优惠券抵扣） | - | F-PM-002 |
| `/api/v1/packages/coupons/my` | GET | 我的优惠券列表（按状态筛选） | - | F-PM-003 |
| `/api/v1/packages/coupons/{id}/receive` | POST | 领取优惠券 | - | F-PM-003 |

### 2.2 后台管理接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/packages/page` | GET | 套餐分页列表 | - | F-PM-008 |
| `/api/v1/packages` | POST | 新增套餐 | package:add | F-PM-009 |
| `/api/v1/packages/{id}/form` | GET | 获取套餐表单数据 | - | F-PM-009 |
| `/api/v1/packages/{id}` | PUT | 修改套餐 | package:edit | F-PM-009 |
| `/api/v1/packages/{id}/status` | PUT | 上架/下架 | package:edit | F-PM-010 |
| `/api/v1/packages/{ids}` | DELETE | 删除套餐 | package:delete | F-PM-001 |
| `/api/v1/packages/sales/stats` | GET | 销售统计 | package:sales | F-PM-012 |
| `/api/v1/packages/coupons/page` | GET | 优惠券分页列表 | - | F-PM-003 |
| `/api/v1/packages/coupons` | POST | 创建优惠券 | package:coupon:add | F-PM-003 |
| `/api/v1/packages/coupons/batch` | POST | 批量发放优惠券 | package:coupon:distribute | F-PM-003 |
| `/api/v1/packages/coupons/{id}` | PUT | 修改优惠券 | package:coupon:edit | F-PM-003 |
| `/api/v1/packages/coupons/{ids}` | DELETE | 删除优惠券 | package:coupon:delete | F-PM-003 |
| `/api/v1/packages/promotions/page` | GET | 促销活动分页列表 | - | F-PM-004 |
| `/api/v1/packages/promotions` | POST | 创建促销活动 | package:promotion:add | F-PM-004 |
| `/api/v1/packages/promotions/{id}` | PUT | 修改促销活动 | package:promotion:edit | F-PM-004 |
| `/api/v1/packages/promotions/{id}/status` | PUT | 促销活动上架/下架 | package:promotion:edit | F-PM-004 |
| `/api/v1/packages/promotions/{id}` | DELETE | 删除促销活动 | package:promotion:delete | F-PM-004 |
| `/api/v1/packages/promotions/{id}/packages` | PUT | 关联套餐（维护活动-套餐关联） | package:promotion:edit | F-PM-004 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| package:add | 新增套餐 |
| package:edit | 编辑/上下架套餐 |
| package:delete | 删除套餐 |
| package:sales | 销售统计 |
| package:coupon:add | 优惠券创建 |
| package:coupon:distribute | 优惠券批量发放 |
| package:coupon:edit | 优惠券编辑 |
| package:coupon:delete | 优惠券删除 |
| package:promotion:add | 促销活动创建 |
| package:promotion:edit | 促销活动编辑/上下架/关联套餐 |
| package:promotion:delete | 促销活动删除 |

> 用户端接口（套餐列表/详情/价格计算/我的优惠券/领取优惠券）与后台查询接口（套餐分页/表单/优惠券列表/促销列表）均为登录态访问。

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| A0401 | 请求资源不存在 | 套餐不存在、优惠券不存在 |
| A0500 | 业务异常 | 促销价高于原价、计费周期非法、套餐名称已被占用、套餐已下架、套餐存在订单无法删除、套餐参与进行中促销无法下架、优惠券已领完、超过限领数量、优惠券不适用该套餐、优惠券状态无效、新用户专享套餐非新用户购买、续费叠加超过最大有效期（1095 天）、体验券每人限领 1 次 |
| A0230 | token无效或已过期 | 未登录访问 |
