# 算法管理模块 API 接口

## 1. 文档概述

本文档定义 **算法管理** 模块的 HTTP API 规范，是该模块 API 契约的唯一权威来源。

- **基础路径**：
  - 算法管理：`/api/v1/algorithms`
  - 模型预测：`/api/v1/prediction`
  - 效果评估：`/api/v1/evaluation`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> 算法导入导出由通用导入导出框架统一调度，路径为 `/api/v1/algorithms`。

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

### 2.2 算法导入/导出接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/_export` | GET | 导出算法元数据（简单查询，同步返回文件流或异步返回任务） | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithms/_export` | POST | 导出算法元数据（复杂查询条件，走异步任务） | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithms/_import` | POST | 导入算法元数据（Excel/CSV，创建算法记录，状态=草稿） | `sys:algorithm:import` | F-M05-007 |
| `/api/v1/algorithms/template` | GET | 下载导入模板（Excel/CSV） | `sys:algorithm:import` | F-M05-007 |

**导出格式**：`.xlsx`（默认）或 `.csv`（通过 `format` 参数指定）

**导入限制**：文件格式 `.xlsx`/`.xls`/`.csv`，文件大小 ≤ 20MB

### 2.3 性能监控接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/{id}/monitor` | GET | 获取算法监控数据（callCount/avgTime/successRate/todayCallCount） | - | F-M05-008 |
| `/api/v1/algorithms/{id}/monitor/stats` | GET | 获取算法统计报表（按天统计，`days` 参数默认 7 天） | - | F-M05-008 |

### 2.4 模型预测接口

> 异步任务模式：POST 立即返回任务 ID 与状态，前端通过 GET 轮询直至终态（1=处理中/2=已完成/3=失败）。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/prediction` | POST | 提交预测任务（异步） | - | F-M05-009 |
| `/api/v1/prediction/{taskId}` | GET | 轮询预测任务状态 | - | F-M05-009 |
| `/api/v1/prediction/logs` | GET | 获取预测日志列表 | - | F-M05-009 |

### 2.5 效果评估接口

> 异步任务模式：与模型预测同理，POST 立即返回任务 ID 与状态，前端通过 GET 轮询直至终态。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/evaluation` | POST | 提交评估任务（异步） | - | F-M05-010 |
| `/api/v1/evaluation/{taskId}` | GET | 轮询评估任务状态 | - | F-M05-010 |
| `/api/v1/evaluation/logs` | GET | 获取评估日志列表 | - | F-M05-010 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `sys:algorithm:add` | 新增算法 |
| `sys:algorithm:edit` | 编辑算法、状态变更（停用/启用/归档） |
| `sys:algorithm:audit` | 审核算法（通过/驳回） |
| `sys:algorithm:delete` | 删除算法 |
| `sys:algorithm:version` | 新增版本、版本回滚 |
| `sys:algorithm:import` | 导入算法、下载模板 |
| `sys:algorithm:export` | 导出算法 |
| - | 登录态接口：查看算法详情、监控数据、版本历史查询、模型预测、效果评估 |

## 4. 业务错误码

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

