# 图像处理模块 API 接口

## 1. 文档概述

本文档定义 **图像处理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：
  - 预测：`/api/v1/prediction`
  - 评估：`/api/v1/evaluation`
  - 参数预设：`/api/v1/presets`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 预测接口

> **异步任务模式**：POST 立即返回 `logId + status=1(处理中)`，前端轮询任务状态接口（2s 间隔）直至完成。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/prediction` | POST | 提交图像处理预测任务（异步） | - | F-M04-001 |
| `/api/v1/prediction/{taskId}` | GET | 轮询任务状态（1=处理中/2=已完成/3=失败） | - | F-M04-001 |
| `/api/v1/prediction/logs` | GET | 预测日志列表（分页，可按算法筛选） | - | F-M04-007 |
| `/api/v1/prediction/batch` | POST | 批量处理（一次提交多张图片，上限按会员等级动态计算） | - | F-M04-002 |
| `/api/v1/prediction/quota` | GET | 查询用户剩余处理次数 | - | F-M04-006 |

**POST `/api/v1/prediction` 请求体（PredictionForm）参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `algorithmId` | Long | ✅ | 算法 ID（任务类型由算法决定，当前为去雾算法） |
| `fileId` | Long | 二选一 | 上传的图片文件 ID |
| `imageUrl` | String | 二选一 | 图片 URL（与 fileId 二选一） |
| `params` | String | 否 | 预测参数（JSON 字符串），由算法解释执行 |
| `recommendedBy` | Long | 否 | 推荐记录 ID（来自推荐管理模块，用于追踪推荐采纳率） |

**POST `/api/v1/prediction` 响应**：

```json
{
  "code": "00000",
  "data": { "logId": 88, "status": 1 }
}
```

**PredictionResultVO 字段**（POST 与 GET `/api/v1/prediction/{taskId}` 返回）：

| 字段 | 类型 | 返回时机 |
|------|------|---------|
| `logId` | Long | POST + GET |
| `status` | Integer | POST + GET（1=处理中/2=已完成/3=失败） |
| `resultUrl` | String | GET status=2 时 |
| `resultThumbnailUrl` | String | GET status=2 时 |
| `time` | Integer | GET status=2/3 时（处理耗时毫秒） |
| `errorMessage` | String | GET status=3 时 |

> **taskType 说明**：当前 `PredictionForm` 无 `taskType` 字段，处理类型由所选算法决定；多任务类型扩展（`params.taskType` 子类型路由）为本次改造目标（见 [需求规格 §1.1](./需求规格.md)）。

### 2.2 评估接口

> **异步任务模式**：同预测模式，POST 立即返回 `logId + status=1(处理中)`。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/evaluation` | POST | 提交指标评估任务（计算 PSNR/SSIM/LPIPS 等指标，异步） | - | 效果评估（能力由[效果对比模块](../效果对比/需求规格.md)承接，F-M05-005） |
| `/api/v1/evaluation/{taskId}` | GET | 轮询评估任务状态 | - | 同上 |
| `/api/v1/evaluation/metrics` | GET | 评估指标历史列表（当前用户） | - | 同上 |
| `/api/v1/evaluation/logs` | GET | 评估日志列表（分页） | - | 同上 |

**POST `/api/v1/evaluation` 请求体（EvaluationForm）参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `algorithmId` | Long | ✅ | 算法 ID |
| `predUrl` | String | ✅ | 预测结果图 URL（待评估图片） |
| `gtUrl` | String | ✅ | 清晰参考图（GT）URL |
| `params` | String | 否 | 评估参数（JSON） |

**POST `/api/v1/evaluation` 响应**：

```json
{ "code": "00000", "data": { "logId": 88, "status": 1 } }
```

**EvaluationResultVO 字段**：`status=2` 时返回 `metrics`（`Map<String,Double>`）与 `time`，`status=3` 时返回 `errorMessage`。

### 2.3 参数预设接口（已实现）

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/presets` | GET | 参数预设列表（系统预设 + 用户自定义，可按算法/类型筛选） | - | F-M04-003 |
| `/api/v1/presets` | POST | 创建自定义预设 | - | F-M04-003 |
| `/api/v1/presets/{id}` | PUT | 更新自定义预设 | - | F-M04-003 |
| `/api/v1/presets/{id}` | DELETE | 删除自定义预设 | - | F-M04-003 |

**PresetForm 参数**：`name`（必填）、`algorithmId`（必填）、`params`（必填，JSON 字符串）、`isDefault`（0/1，可选）。

> **说明**：预设类型分为 `system`（系统预设，只读，不可修改/删除）与 `custom`（用户自定义，仅本人可操作）。系统预设由管理员维护（见 [算法管理模块](../算法管理/API接口.md)）。

### 2.4 配额接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/prediction/quota` | GET | 查询用户剩余处理次数 | - | F-M04-006 |

**PredictionQuotaVO 字段**：`remaining`（剩余次数）、`total`（总次数，按会员等级权益）、`used`（已用次数）、`resetDate`（下月 1 日重置日期）。

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 图像处理无特殊权限标识，登录用户即可操作；VIP 配额通过会员等级控制 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 轮询的 taskId 不存在、预设不存在、算法不存在 |
| `A0500` | 业务异常 | 批量处理图片数量超过会员等级上限；图片来源为空 |
| `A0501` | 数据已存在 | 预设名称冲突（如系统预设同名） |
| `A0502` | 数据状态不允许 | 尝试修改/删除系统预设、操作他人的自定义预设 |
| `A0515` | 当月次数已用完 | 配额耗尽（前端引导升级 VIP） |
| `A0230` | token无效或已过期 | 未登录访问 |

> 预测任务执行失败不通过错误码返回——任务状态通过 `status=3(失败)` + `errorMessage` 表达；Python 算法轮询超时（300s）时任务标记为失败。
