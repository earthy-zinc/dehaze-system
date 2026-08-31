# AI计费管理模块 - API接口

## 1. 文档概述

本文档定义 **AI计费管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/ai-billing`

> **模块边界**：本模块 API 聚焦积分账户（余额/流水/到账/回退）、计量计费（计费明细/账单/退款申请）与成本核算（成本单价/成本-利润/账单对账）。积分卡商品浏览与购买、积分卡支付与售后审核走套餐管理/订单管理模块，不在本模块定义。

## 2. 接口清单

### 2.1 用户端接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai-billing/balance` | GET | 余额查询（返回余额、配额使用情况）；可选 `userId` 参数，管理员下钻指定用户 | - / `ai:billing:stat`* | F-MB-001, F-MB-005 |
| `/api/v1/ai-billing/summary` | GET | 消耗汇总查询（日/月消耗趋势、模型分布、节省汇总）；可选 `dimension` 参数（`day`/`month`，默认 `day`） | - | F-MB-009 |
| `/api/v1/ai-billing/records` | GET | 计费明细查询（分页查询计费记录，含申诉状态）；可选 `userId` 参数，管理员下钻指定用户 | - / `ai:billing:stat`* | F-MB-009 |
| `/api/v1/ai-billing/credit-logs` | GET | 流水查询（分页查询余额变动流水）；可选 `userId` 参数，管理员下钻指定用户 | - / `ai:billing:stat`* | F-MB-001 |

> *`userId` 参数语义：不传或等于当前登录用户 ID 时查询本人（登录态即可）；传入**他人** `userId` 时需 `ai:billing:stat` 权限（ROOT 直接放行），用于管理端下钻用户余额/明细/流水。`userId` 非法值（<1）返回参数错误。
| `/api/v1/ai-billing/bills/{month}` | GET | 账单查询（查询月结账单） | - | F-MB-009 |
| `/api/v1/ai-billing/bills/{month}/download` | GET | 账单下载（下载月结账单） | - | F-MB-009 |
| `/api/v1/ai-billing/refunds` | POST | 退款申请（用户申请误扣退款） | - | F-MB-003 |

### 2.2 管理员接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai-billing/stats` | GET | 管理员统计查询（按多维度统计） | ai:billing:stat | F-MB-009 |
| `/api/v1/ai-billing/anomalies` | GET | 异常计费记录查询（异常清单与趋势） | ai:billing:stat | F-MB-007 |
| `/api/v1/ai-billing/adjust` | POST | 管理员手动调整（手动调整用户积分） | ai:billing:adjust | F-MB-002 |
| `/api/v1/ai-billing/refunds` | GET | 退款申请列表（管理端审核中心：分页 + 状态/用户/时间筛选） | ai:billing:refund | F-MB-003 |
| `/api/v1/ai-billing/refunds/{id}/audit` | POST | 管理员退款审核（审核误扣退款申请） | ai:billing:refund | F-MB-003 |
| `/api/v1/ai-billing/costs` | GET | 成本单价列表（按模型/供应商/版本查询） | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs` | POST | 新增成本单价（供应商调价生成新价格版本） | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs/{id}` | PUT | 更新成本单价 | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs/{id}` | DELETE | 停用/删除成本单价 | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/cost-stats` | GET | 成本-利润统计（收入/成本/毛利，按模型/供应商/时间） | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/reconcile/import` | POST | 供应商账单导入（实际账单对账） | ai:billing:cost | F-MB-010 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| ai:billing:stat | 管理员计费统计查询；`balance`/`records`/`credit-logs` 指定 `userId` 查询他人数据（管理端下钻用户详情） |
| ai:billing:adjust | 管理员手动调整用户积分 |
| ai:billing:refund | 管理员退款审核（误扣补偿申请） |
| ai:billing:cost | 模型成本单价维护与成本-利润统计（成本数据仅管理员可见） |

> 用户侧查询接口（余额/计费明细/流水/账单查询/账单下载/退款申请）均为登录态访问，不传 `userId` 时仅可查询/操作本人数据；管理员凭 `ai:billing:stat` 可指定 `userId` 下钻任意用户。本模块退款申请接口仅面向**误扣补偿**场景。

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| A0400 | 用户请求参数错误 | 退款申请 `amount` 不合法（非正数或超过可退金额）；管理员调整积分为 0 |
| A0401 | 资源不存在 | 账单月份不存在/无该月记录；账单/计费记录不存在 |
| A0403 | 无权限 | 访问他人余额/账单/记录 |
| A0680 | 退款申请已存在 | 同一计费记录重复申请退款 |
| A0681 | 退款审核失败 | 原计费记录已退款或不存在 |
| A0682 | 配额不足/欠费熔断 | 配额超限或欠费状态调用 AI 能力 |
