# 订单管理模块 - API接口

## 1. 文档概述

本文档定义 **订单管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/orders`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

**基础路径**：`/api/v1/orders`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders` | POST | 创建订单 | - |
| `/api/v1/orders/my` | GET | 我的订单列表 | - |
| `/api/v1/orders/{orderNo}` | GET | 订单详情 | - |
| `/api/v1/orders/{orderNo}/cancel` | PUT | 取消订单 | - |
| `/api/v1/orders/{orderNo}/pay` | POST | 发起支付 | - |
| `/api/v1/orders/{orderNo}/refund` | POST | 申请退款 | - |
| `/api/v1/orders/auto-renew/config` | PUT | 修改自动续费设置 | - |

### 2.2 支付回调接口

**基础路径**：`/api/v1/orders/payment`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders/payment/wechat/callback` | POST | 微信支付回调 | - |
| `/api/v1/orders/payment/alipay/callback` | POST | 支付宝回调 | - |

### 2.3 后台管理接口

**基础路径**：`/api/v1/orders`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders/page` | GET | 订单分页列表 | - |
| `/api/v1/orders/{orderNo}` | GET | 订单详情 | - |
| `/api/v1/orders/refunds/page` | GET | 退款审核列表 | - |
| `/api/v1/orders/refunds/{id}/approve` | PUT | 退款审核通过 | order:refund:approve |
| `/api/v1/orders/refunds/{id}/reject` | PUT | 退款审核驳回 | order:refund:approve |
| `/api/v1/orders/stats` | GET | 订单统计 | - |

> **导入导出接口**：订单模块的导出（`GET/POST /api/v1/orders/_export`）、导入（`POST /api/v1/orders/_import`）、模板下载（`GET /api/v1/orders/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §8.2 通用CRUD接口模板。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| order:refund:approve | 退款审核 | 按钮显示 + 接口校验 |

## 4. 状态枚举

### 4.1 订单状态

| 状态值 | 显示 | 说明 |
|--------|------|------|
| pending | 待支付（橙色标签） | 订单已创建，等待支付 |
| paid | 已支付（蓝色标签） | 支付成功，权益已生效 |
| completed | 已完成（灰色标签） | 套餐到期，订单归档 |
| cancelled | 已取消（灰色标签） | 用户取消或超时自动取消 |
| refunding | 退款中（黄色标签） | 退款申请待审核 |
| refunded | 已退款（灰色标签） | 退款完成，权益已回收 |

### 4.2 支付方式

| 值 | 说明 |
|----|------|
| wechat | 微信支付 |
| alipay | 支付宝 |
| balance | 平台余额 |
| combined | 组合支付（余额 + 第三方） |

### 4.3 退款状态

| 状态值 | 说明 |
|--------|------|
| refunding | 退款请求已发送，等待审核 |
| refunded | 退款成功 |
| refund_failed | 退款失败，需人工介入 |

## 5. 业务错误码

| 错误码 | 错误信息 | 触发场景 |
|-------|---------|---------|
| ORDER_NOT_FOUND | 订单不存在 | 查询/操作时订单不存在 |
| ORDER_STATUS_INVALID | 订单状态不允许此操作 | 非待支付订单发起支付、非已支付订单申请退款等 |
| ORDER_EXPIRED | 订单已超时 | 超过 30 分钟未支付，订单已自动取消 |
| ORDER_ALREADY_PAID | 订单已支付 | 重复支付请求 |
| REFUND_TIME_EXCEEDED | 超过退款时限 | 支付成功超过 7 天申请退款 |
| REFUND_USAGE_EXCEEDED | 权益使用超限 | 已用次数超过总配额 50% |
| REFUND_NOT_SUPPORTED | 该套餐不支持退款 | 限时特价套餐退款 |
| PAYMENT_AMOUNT_MISMATCH | 支付金额与订单金额不一致 | 回调金额校验失败 |
| DUPLICATE_ORDER | 短时间内重复下单 | 同一用户同一套餐 5 秒内重复下单 |
