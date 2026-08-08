# 订单管理模块 - API接口

## 1. 文档概述

本文档定义 **订单管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/orders`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。
> 套餐定义/价格计算/优惠券模板接口见[套餐管理模块 API接口](../套餐管理/API接口.md)。

## 2. 接口清单

### 2.1 用户端接口

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders` | POST | 创建订单（校验套餐/价格/锁定优惠券） | - |
| `/api/v1/orders/my` | GET | 我的订单列表（按状态筛选） | - |
| `/api/v1/orders/{orderNo}` | GET | 订单详情（含支付流水、退款记录） | - |
| `/api/v1/orders/{orderNo}/cancel` | PUT | 取消订单（仅待支付，需取消原因） | - |
| `/api/v1/orders/{orderNo}/pay` | POST | 发起支付（微信/支付宝返回支付参数；余额直接生效） | - |
| `/api/v1/orders/{orderNo}/refund` | POST | 申请退款（reason + customReason） | - |
| `/api/v1/orders/auto-renew/config` | PUT | 开启/关闭自动续费（upsert 配置） | - |
| `/api/v1/orders/auto-renew/config` | GET | 查询自动续费配置 | - |

### 2.2 支付回调接口

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders/payment/wechat/callback` | POST | 微信支付回调（验签+幂等） | - |
| `/api/v1/orders/payment/alipay/callback` | POST | 支付宝回调 | - |

### 2.3 后台管理接口

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/orders/page` | GET | 订单分页列表 | - |
| `/api/v1/orders/{orderNo}` | GET | 订单详情 | - |
| `/api/v1/orders/refunds/page` | GET | 退款审核列表 | - |
| `/api/v1/orders/refunds/{refundId}/approve` | PUT | 退款审核通过 | order:refund:approve |
| `/api/v1/orders/refunds/{refundId}/reject` | PUT | 退款审核驳回 | order:refund:approve |
| `/api/v1/orders/stats` | GET | 订单统计（总额/状态/支付方式/套餐/每日趋势） | - |

> **导入导出接口**：订单模块的导出（`GET/POST /api/v1/orders/_export`）、导入（`POST /api/v1/orders/_import`）、模板下载（`GET /api/v1/orders/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §7 通用CRUD接口模板。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| order:refund:approve | 退款审核通过/驳回 | 按钮显示 + 接口校验 |
| order:stats | 订单统计 | 前端按钮控制（`v-hasPerm`），后端接口无权限校验 |

## 4. 状态枚举

### 4.1 订单状态

| 状态码 | 状态值 | 说明 |
|--------|--------|------|
| 1 | pending | 待支付：订单已创建，等待支付 |
| 2 | paid | 已支付：支付成功，权益已生效 |
| 3 | completed | 已完成：套餐到期，订单归档 |
| 4 | cancelled | 已取消：用户取消或超时自动取消 |
| 5 | refunding | 退款中：退款申请待审核 |
| 6 | refunded | 已退款：退款完成 |

### 4.2 支付方式

| 值 | 说明 |
|----|------|
| wechat | 微信支付 |
| alipay | 支付宝 |
| balance | 平台余额 |
| combined | 组合支付（**当前无渠道实现，发起支付报错**） |

### 4.3 退款状态

| 状态码 | 状态值 | 说明 |
|--------|--------|------|
| 1 | refunding | 退款中：退款申请待审核 |
| 2 | refunded | 退款成功 |
| 3 | refund_failed | 退款失败：渠道退款失败或审核驳回，需人工介入/自动重试 |

## 5. 业务错误码

| 错误码 | 错误信息 | 触发场景 | 状态 |
|-------|---------|---------|------|
| ORDER_NOT_FOUND | 订单不存在 | 查询/操作时订单不存在或非本人订单 | ✅ 使用中 |
| ORDER_STATUS_INVALID | 订单状态不允许此操作 | 非待支付订单发起支付/取消；非已支付订单申请退款等 | ✅ 使用中 |
| ORDER_EXPIRED | 订单已超时 | 超过 30 分钟未支付后发起支付 | ✅ 使用中 |
| ORDER_ALREADY_PAID | 订单已支付 | 重复支付请求 | ⚠️ 已定义未使用（状态校验统一返回 ORDER_STATUS_INVALID） |
| REFUND_TIME_EXCEEDED | 超过退款时限 | 支付成功超过 7 天申请退款 | ⚠️ 仅 Python 端使用（Java 端未校验） |
| REFUND_USAGE_EXCEEDED | 权益使用超限 | 已用次数超过总配额 50% | ❌ 已定义未使用（无此校验） |
| REFUND_NOT_SUPPORTED | 该套餐不支持退款 | 限时特价套餐退款 | ❌ 已定义未使用（无此校验） |
| REFUND_ALREADY_EXISTS | 该订单已存在退款申请 | 重复申请退款 | ✅ 使用中 |
| PAYMENT_AMOUNT_MISMATCH | 支付金额与订单金额不一致 | 回调金额校验失败 | ⚠️ 仅 Python 端抛异常（Java 端返回失败+日志） |
| DUPLICATE_ORDER | 短时间内重复下单 | 同一用户同一套餐 5 秒内重复下单 | ✅ 使用中 |
| CALL_THIRD_PARTY_SERVICE_ERROR | 调用第三方服务出错 | 渠道统一下单/代扣失败 | ✅ 使用中 |
| PARAM_ERROR | 参数错误 | 不支持的支付方式/渠道 | ✅ 使用中 |
| COUPON_ALREADY_USED | 优惠券已被使用 | 下单锁定优惠券时状态非未使用 | ✅ 使用中 |

> **说明**：`REFUND_USAGE_EXCEEDED`/`REFUND_NOT_SUPPORTED` 为预留错误码（退款条件校验待实现）；`REFUND_TIME_EXCEEDED`/`PAYMENT_AMOUNT_MISMATCH` 仅 Python 端使用，Java/Python 行为待统一（见 [需求规格 §9](../../订单管理/需求规格.md)）。
