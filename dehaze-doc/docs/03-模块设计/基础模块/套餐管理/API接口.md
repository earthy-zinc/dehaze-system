# 套餐管理模块 - API接口

## 1. 文档概述

本文档定义 **套餐管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/packages`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/packages`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/packages` | GET | 在售套餐列表 | - |
| `/api/v1/packages/{id}` | GET | 套餐详情 | - |
| `/api/v1/packages/calculate-price` | GET | 价格计算（下单前预览） | - |
| `/api/v1/packages/coupons/my` | GET | 我的优惠券列表 | - |
| `/api/v1/packages/coupons/{id}/receive` | POST | 领取优惠券 | - |

### 2.2 后台管理接口

**基础路径**：`/api/v1/packages`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/packages/page` | GET | 套餐分页列表 | - |
| `/api/v1/packages` | POST | 新增套餐 | package:add |
| `/api/v1/packages/{id}/form` | GET | 获取套餐表单数据 | - |
| `/api/v1/packages/{id}` | PUT | 修改套餐 | package:edit |
| `/api/v1/packages/{id}/status` | PUT | 上架/下架 | package:edit |
| `/api/v1/packages/{ids}` | DELETE | 删除套餐 | package:delete |
| `/api/v1/packages/coupons/page` | GET | 优惠券分页列表 | - |
| `/api/v1/packages/coupons` | POST | 创建优惠券 | package:coupon:add |
| `/api/v1/packages/coupons/batch` | POST | 批量发放优惠券 | package:coupon:distribute |
| `/api/v1/packages/coupons/{id}` | PUT | 修改优惠券 | package:coupon:edit |
| `/api/v1/packages/coupons/{ids}` | DELETE | 删除优惠券 | package:coupon:delete |
| `/api/v1/packages/sales/stats` | GET | 销售统计 | package:sales |

> **导入导出接口**：套餐模块的导出（`GET/POST /api/v1/packages/_export`）、导入（`POST /api/v1/packages/_import`）、模板下载（`GET /api/v1/packages/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §8.2 通用CRUD接口模板。

### 2.3 价格计算接口

**路径**：`GET /api/v1/packages/calculate-price`

**功能**：下单前预览价格明细，根据套餐促销活动和优惠券计算应付金额。后端重新计算，不信任前端传入金额。

**请求参数**（Query）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| packageId | Long | ✅ | 套餐ID |
| userCouponId | Long | ❌ | 用户优惠券实例ID，不传则不使用优惠券 |

**响应结构**：`Result<PriceResult>`

| 字段 | 类型 | 说明 |
|------|------|------|
| originalPrice | Long | 原价（分） |
| discountAmount | Long | 促销折扣金额（分） |
| couponAmount | Long | 优惠券抵扣金额（分） |
| payableAmount | Long | 应付金额（分） |
| promotion | PromotionVO | 命中的促销活动信息，无活动时为 null |

**业务规则**：
- 促销折扣取该套餐所有进行中活动的最大折扣值
- 优惠券计算基数为 `salePrice - discountAmount`
- `payableAmount = salePrice - discountAmount - couponAmount`，最小为 0
- 优惠券状态必须为 1（未使用）或 4（已锁定），且未过期

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| package:add | 新增套餐 | 按钮显示 + 接口校验 |
| package:edit | 编辑套餐 | 按钮显示 + 接口校验 |
| package:delete | 删除套餐 | 按钮显示 + 接口校验 |
| package:sales | 销售统计 | 按钮显示 + 接口校验 |
| package:coupon:add | 优惠券创建 | 按钮显示 + 接口校验 |
| package:coupon:distribute | 优惠券批量发放 | 按钮显示 + 接口校验 |
| package:coupon:edit | 优惠券编辑 | 按钮显示 + 接口校验 |
| package:coupon:delete | 优惠券删除 | 按钮显示 + 接口校验 |

## 4. 状态枚举

### 4.1 套餐状态

| 状态值 | 显示 | 说明 |
|--------|------|------|
| 1 | 在售（绿色标签） | 用户端可见，可购买 |
| 0 | 下架（灰色标签） | 用户端不可见，不可购买 |

### 4.2 优惠券状态

| 状态值 | 显示 | 说明 |
|--------|------|------|
| 1 | 未使用（unused） | 优惠券已领取，待使用 |
| 2 | 已使用（used） | 优惠券已在订单中核销 |
| 3 | 已过期（expired） | 优惠券超过有效期 |
| 4 | 已锁定（locked） | 优惠券已被订单锁定，待支付 |

### 4.3 计费周期

| 值 | 说明 |
|----|------|
| monthly | 月卡（30 天） |
| quarterly | 季卡（90 天） |
| yearly | 年卡（365 天） |

## 5. 业务错误码

| 错误码 | 错误信息 | 触发场景 |
|-------|---------|---------|
| PACKAGE_NOT_FOUND | 套餐不存在 | 查询/编辑时套餐不存在 |
| PACKAGE_OFF_SHELF | 套餐已下架 | 用户购买已下架套餐 |
| PACKAGE_HAS_ORDERS | 套餐下已有关联订单，无法删除 | 删除时套餐存在关联订单 |
| COUPON_NOT_FOUND | 优惠券不存在 | 查询/使用时优惠券不存在 |
| COUPON_EXPIRED | 优惠券已过期 | 使用过期优惠券 |
| COUPON_ALREADY_USED | 优惠券已使用 | 重复使用优惠券 |
| COUPON_STOCK_EMPTY | 优惠券已领完 | 领取时库存为 0 |
| COUPON_NOT_APPLICABLE | 优惠券不适用于该套餐 | 优惠券限定套餐与当前套餐不匹配 |
