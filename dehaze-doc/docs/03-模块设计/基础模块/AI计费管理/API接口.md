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
| `/api/v1/ai-billing/bills/{month}` | GET | 账单查询（查询月结账单；非当前月份且该月无任何消费/充值/退款记录时返回 A0401"账单不存在"，当前月份无记录属正常，返回全 0 账单） | - | F-MB-009 |
| `/api/v1/ai-billing/bills/{month}/download` | GET | 账单下载（返回 `{code,msg,data}` JSON 信封，data 为账单内容，前端可另存为文件；月份行为同账单查询） | - | F-MB-009 |
| `/api/v1/ai-billing/refunds` | POST | 退款申请（用户申请误扣退款） | - | F-MB-003 |

### 2.2 管理员接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/ai-billing/stats` | GET | 管理员统计查询（按多维度统计；`groupBy` ∈ `user`/`model`/`billType`/`day`，默认 `model`，非法值返回 `A0400`；`userId`/`modelId`/`billType`/`dateStart`/`dateEnd` 为过滤条件） | ai:billing:stat | F-MB-009 |
| `/api/v1/ai-billing/anomalies` | GET | 异常计费记录查询（异常清单与趋势） | ai:billing:stat | F-MB-007 |
| `/api/v1/ai-billing/adjust` | POST | 管理员手动调整（手动调整用户积分） | ai:billing:adjust | F-MB-002 |
| `/api/v1/ai-billing/refunds` | GET | 退款申请列表（管理端审核中心：分页 + 状态/用户/时间筛选） | ai:billing:refund | F-MB-003 |
| `/api/v1/ai-billing/refunds/{id}/audit` | POST | 管理员退款审核（审核误扣退款申请） | ai:billing:refund | F-MB-003 |
| `/api/v1/ai-billing/costs` | GET | 成本单价列表（按模型/供应商/版本查询） | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs` | POST | 新增成本单价（供应商调价生成新价格版本） | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs/{id}` | PUT | 更新成本单价 | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/costs/{id}` | DELETE | 停用/删除成本单价 | ai:billing:cost | F-MB-010 |
| `/api/v1/ai-billing/cost-stats` | GET | 成本-利润统计（groupBy=overall 时为整体双口径毛利；model/provider 时为成本分组分解） | ai:billing:cost | F-MB-010 |
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
| A0400 | 用户请求参数错误 | 退款申请 `amount` 不合法（非正数或超过可退金额）；管理员调整积分为 0；`dimension`/`groupBy`/时间格式等参数不合法 |
| A0401 | 资源不存在 | 账单月份不存在/无该月记录；账单/计费记录不存在 |
| A0301 | 访问未授权 | 访问他人余额/账单/记录；管理员接口缺对应权限码 |
| A0680 | 退款申请已存在 | 同一计费记录重复申请退款 |
| A0681 | 退款审核失败 | 原计费记录已退款或不存在 |
| A0682 | 配额不足/欠费熔断 | 配额超限、欠费状态或权益数据缺失/停用（fail-closed）时调用 AI 能力 |

## 5. 关键接口契约

### 5.1 异常计费记录查询 GET /anomalies

需 `ai:billing:stat` 权限。分页查询（`pageNum`/`pageSize`），筛选参数：`userId`、`anomalyType`、`status`、`dateStart`、`dateEnd`（格式 `yyyy-MM-dd` 或 `yyyy-MM-dd HH:mm:ss`）。

响应 `data.list[]` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 主键 |
| userId | int | 用户 ID |
| billingId | int \| null | 关联计费记录 ID（consecutive_quota_fail 类无关联记录为 null） |
| anomalyType | string | 异常规则类型（见下表四类） |
| detail | string | 异常详情描述（含触发数值） |
| status | int | 处理状态三态：0-待处理 / 1-已处理 / 2-已忽略 |
| triggerAt | string | 触发时间（Asia/Shanghai） |

异常规则类型（四类）：

| anomalyType | 触发条件 |
|-------------|---------|
| `single_high` | 单次消耗 > 月限额 × 10% |
| `burst` | 5 分钟窗口内累计消耗 > 日限额 × 50% |
| `consecutive_quota_fail` | 24 小时内配额不足触发 ≥ 10 次 |
| `empty_high_output` | 输出 0 token 但输入 > 10000 token |

### 5.2 成本-利润统计 GET /cost-stats

需 `ai:billing:cost` 权限。参数：`startTime`/`endTime`（时间范围）、`groupBy`（`overall` 默认 / `model` / `provider`）、`modelId`/`providerId`（成本过滤）。

- `groupBy=overall`（默认）：返回 2 行整体双口径，字段 `metric`（overall-官方口径 / ai-AI 参考口径）+ `revenue`/`cost`/`profit`/`profitRate`
- `groupBy=model`/`provider`：返回分组行，仅含 `dimension`（模型标识/供应商 ID）+ `cost`——订单实收无法按模型/供应商归因，分组行不返回收入/毛利/口径字段

### 5.3 消耗汇总 GET /summary（计量口径）

仅统计 bill_type=chat 的记录；asr/tts 类型记录的 input_tokens 存储秒数/字符数（非 token），不纳入 token/节省类统计。
