# 算法管理模块 API 接口

## 1. 文档概述

本文档定义 **算法管理** 模块的 HTTP API 规范,是该模块 API 契约的**唯一权威来源**。

- **基础路径**:
  - 算法管理: `/api/v1/algorithms`
  - 算法导入导出: `/api/v1/algorithm`（通用导入导出框架，模块名 `algorithm` 为单数）
  - 模型预测: `/api/v1/prediction`
  - 效果评估: `/api/v1/evaluation`
- **公共约定**: 参见 [../../02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**: [需求规格.md](./需求规格.md)
- **后端实现**: [后端实现.md](./后端实现.md)

> **重要**: 接口详细参数/响应结构可通过 API 文档 MCP 查询,本文档仅定义接口清单和权限标识。

## 2. 接口清单

### 2.1 算法管理接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms` | GET | 获取算法树形表格 | - | F-M05-001 |
| `/api/v1/algorithms/{id}` | GET | 根据ID获取算法详情 | - | F-M05-002 |
| `/api/v1/algorithms/options` | GET | 获取算法下拉选项 | - | F-M05-001 |
| `/api/v1/algorithms/list` | GET | 获取所有算法扁平列表（不分页，不构建树形） | - | F-M05-001 |
| `/api/v1/algorithms` | POST | 新增算法 | `sys:algorithm:add` | F-M05-003 |
| `/api/v1/algorithms/{id}` | PUT | 修改算法 | `sys:algorithm:edit` | F-M05-005 |
| `/api/v1/algorithms/{id}/status` | PUT | 修改算法状态 | `sys:algorithm:edit` | F-M05-006 |
| `/api/v1/algorithms/{id}/audit` | PUT | 审核算法（通过/驳回） | `sys:algorithm:audit` | F-M05-004 |
| `/api/v1/algorithms` | DELETE | 删除算法（批量，`ids` 查询参数，支持单个/多个） | `sys:algorithm:delete` | F-M05-006 |
| `/api/v1/algorithms/{id}/version` | POST | 新增算法版本 | `sys:algorithm:version` | F-M05-005 |
| `/api/v1/algorithms/{id}/versions` | GET | 获取算法版本历史 | - | F-M05-005 |
| `/api/v1/algorithms/{id}/rollback` | POST | 版本回滚（`versionId` 查询参数） | `sys:algorithm:version` | F-M05-005 |

> 删除接口说明：系统仅提供批量删除接口 `DELETE /api/v1/algorithms?ids=1,2,3`，单个删除同样通过传入单个 id 实现。删除时递归收集子孙算法一并删除。

### 2.2 算法导入/导出接口

> 算法导入导出由**通用导入导出框架**统一调度，导出/导入范围为算法**元数据**（Excel/CSV），**不含模型权重文件**。模型权重文件托管在 nginx-dataset 静态服务，其上传/下载通过算法详情接口单独处理。模块名 `algorithm` 为单数，与算法 CRUD 路径 `/api/v1/algorithms`（复数）区分。通用框架详见 [任务管理/后端实现.md](../../基础模块/任务管理/后端实现.md)。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithm/_export` | GET | 导出算法元数据（简单查询，同步返回文件流或异步返回任务） | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithm/_export` | POST | 导出算法元数据（复杂查询条件，走异步任务） | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithm/_import` | POST | 导入算法元数据（Excel/CSV，创建算法记录，状态=草稿） | `sys:algorithm:import` | F-M05-007 |
| `/api/v1/algorithm/template` | GET | 下载导入模板（Excel/CSV） | `sys:algorithm:import` | F-M05-007 |

**导出格式**：`.xlsx`（默认）或 `.csv`（通过 `format` 参数指定）

**导入限制**：文件格式 `.xlsx`/`.xls`/`.csv`，文件大小 ≤ 20MB

**导入校验规则**：

| 校验项 | 规则 | 错误码 |
|--------|------|--------|
| 文件格式 | 必须为 Excel/CSV | A0701 |
| 必填字段 | name、type 不能为空 | A0706 |
| 名称唯一 | 算法名称不能与已有算法重复 | A0707 |
| 字段格式 | type 必须为合法算法类型 | A0707 |

### 2.3 性能监控接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/{id}/monitor` | GET | 获取算法监控数据 | - | F-M05-008 |
| `/api/v1/algorithms/{id}/monitor/stats` | GET | 获取算法统计报表（当前复用监控数据接口） | - | F-M05-008 |

### 2.4 模型预测接口

> **异步任务模式**：POST 立即返回 `{ logId, status: "processing" }`，前端通过 GET 轮询 `status` 字段直至 `completed` 或 `failed`。详见 [API 规范 §8.3](../../../02-系统架构/04-API规范.md#83-预测评估异步任务接口)。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/prediction` | POST | 提交预测任务，立即返回 logId + status=processing | - | F-M05-009 |
| `/api/v1/prediction/{taskId}` | GET | 轮询预测任务状态（processing/completed/failed） | - | F-M05-009 |
| `/api/v1/prediction/logs` | GET | 获取预测日志列表 | - | F-M05-009 |

**POST 响应**（`PredictionResultVO`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `logId` | Long | 预测日志 ID |
| `status` | int | 任务状态：`1=处理中(processing)` / `2=已完成(completed)` / `3=失败(failed)` |
| `resultUrl` | String | 结果图 URL（GET 轮询 completed 时返回） |
| `resultThumbnailUrl` | String | 结果缩略图 URL（GET 轮询 completed 时返回） |
| `time` | int | 处理耗时（毫秒，GET 轮询 completed/failed 时返回） |
| `errorMessage` | String | 失败错误信息（GET 轮询 failed 时返回） |

> 缓存命中时 POST 直接返回 `status=2(completed)` + 完整结果，无需轮询。

### 2.5 效果评估接口

> **异步任务模式**：与模型预测同理，POST 立即返回 `{ logId, status: "processing" }`，前端通过 GET 轮询直至终态。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/evaluation` | POST | 提交评估任务，立即返回 logId + status=processing | - | F-M05-010 |
| `/api/v1/evaluation/{taskId}` | GET | 轮询评估任务状态（processing/completed/failed） | - | F-M05-010 |
| `/api/v1/evaluation/logs` | GET | 获取评估日志列表 | - | F-M05-010 |

**POST 响应**（`EvaluationResultVO`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `logId` | Long | 评估日志 ID |
| `status` | int | 任务状态：`1=处理中(processing)` / `2=已完成(completed)` / `3=失败(failed)` |
| `metrics` | Map<String,Double> | 评估指标（GET 轮询 completed 时返回，如 `{"PSNR": 28.56, "SSIM": 0.92}`） |
| `time` | int | 处理耗时（毫秒，GET 轮询 completed/failed 时返回） |
| `errorMessage` | String | 失败错误信息（GET 轮询 failed 时返回） |

## 3. 权限标识汇总

| 权限标识 | 说明 | 控制范围 |
|---------|------|---------|
| `sys:algorithm:add` | 新增算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:edit` | 编辑算法/修改状态 | 按钮显示 + 接口校验 |
| `sys:algorithm:audit` | 审核算法（通过/驳回） | 按钮显示 + 接口校验 |
| `sys:algorithm:stop` | 停用/启用算法 | 按钮显示 + 接口校验（复用 edit 权限） |
| `sys:algorithm:delete` | 删除算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:view` | 查看算法 | 默认所有用户 |
| `sys:algorithm:version` | 版本管理 | 按钮显示 + 接口校验 |
| `sys:algorithm:import` | 导入算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:export` | 导出算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:monitor` | 性能监控 | 前端按钮显示控制（接口层当前未强制校验） |

> **说明**：监控接口 `/{id}/monitor`、`/{id}/monitor/stats` 及版本历史查询 `/{id}/versions` 当前未加 `@PreAuthorize` 注解，登录用户即可访问。`sys:algorithm:monitor` 权限标识主要用于前端按钮显示控制。

## 4. 业务错误码

算法模块复用系统通用错误码体系（A 系列），不单独定义 B02xx 算法专属错误码。B0200/B0210/B0220 等为系统级错误码（容灾/限流/降级），与算法业务无关。

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 查询/编辑/删除不存在的算法 |
| `A0501` | 数据已存在 | 新增/导入时算法名称重复 |
| `A0502` | 数据状态不允许 | 当前状态下不能执行该操作（如非待审核状态审核、无效状态值） |
| `A0504` | 存在关联数据，无法删除 | 算法仍有进行中的预测任务 |
| `A0701` | 文件格式不支持 | 导入文件非 Excel/CSV 格式 |
| `A0706` | 必填字段为空 | 导入时 name、type 为空 |
| `A0707` | 数据校验失败 | 导入时名称重复、类型不合法 |
| `B0001` | 系统执行出错 | 通用业务异常（如驳回原因未填写、状态流转非法） |
| `A0230` | token无效或已过期 | 未认证访问 |

> **审核驳回**：驳回时未填写备注，后端抛出 `BusinessException("驳回时必须填写审核备注")`，错误码为 `B0001`。

## 5. 关键接口参数说明

### 5.1 审核算法（`PUT /api/v1/algorithms/{id}/audit`）

请求体 `AlgorithmAuditForm`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `approved` | Boolean | ✅ | `true`=审核通过，`false`=审核驳回 |
| `remark` | String | 驳回时必填 | 审核备注/驳回原因 |

### 5.2 修改算法状态（`PUT /api/v1/algorithms/{id}/status`）

请求体：`{"status": <Integer>}`，状态值参见 [需求规格 §2.1.3](./需求规格.md#213-算法状态)。

### 5.3 版本回滚（`POST /api/v1/algorithms/{id}/rollback`）

查询参数：`versionId=<Long>`（目标版本 ID）。

## 6. 接口详情查询

> 接口的详细请求参数、响应结构、Schema 定义可通过以下方式获取:
>
> 1. **API 文档 MCP**: 调用 `read_project_oas_wht4eg` 获取 OpenAPI Spec
> 2. **Swagger UI**: 访问 `/swagger-ui/index.html`(开发环境)

---

**文档版本**: v1.2.0
**最后更新**: 2026-07-30
**维护者**: 技术文档团队
