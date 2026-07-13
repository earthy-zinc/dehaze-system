# 算法管理模块 API 接口

## 1. 文档概述

本文档定义 **算法管理** 模块的 HTTP API 规范,是该模块 API 契约的**唯一权威来源**。

- **基础路径**:
  - 算法管理: `/api/v1/algorithms`
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
| `/api/v1/algorithms` | POST | 新增算法 | `sys:algorithm:add` | F-M05-003 |
| `/api/v1/algorithms/{id}` | PUT | 修改算法 | `sys:algorithm:edit` | F-M05-005 |
| `/api/v1/algorithms/{id}/status` | PUT | 修改算法状态 | `sys:algorithm:edit` | F-M05-006 |
| `/api/v1/algorithms/{id}/audit` | PUT | 审核算法（通过/驳回） | `sys:algorithm:audit` | F-M05-004 |
| `/api/v1/algorithms/{id}` | DELETE | 删除单个算法 | `sys:algorithm:delete` | F-M05-006 |
| `/api/v1/algorithms` | DELETE | 批量删除算法 | `sys:algorithm:delete` | F-M05-006 |
| `/api/v1/algorithms/{id}/version` | POST | 新增算法版本 | `sys:algorithm:version` | F-M05-005 |
| `/api/v1/algorithms/{id}/versions` | GET | 获取算法版本历史 | - | F-M05-005 |
| `/api/v1/algorithms/{id}/rollback` | POST | 版本回滚 | `sys:algorithm:version` | F-M05-005 |

### 2.2 算法导入/导出接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/{id}/_export` | GET | 导出单个算法（配置+模型） | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithms/_export` | POST | 批量导出算法 | `sys:algorithm:export` | F-M05-007 |
| `/api/v1/algorithms/_import` | POST | 导入算法包 | `sys:algorithm:import` | F-M05-007 |
| `/api/v1/algorithms/_import/validate` | POST | 校验导入包 | `sys:algorithm:import` | F-M05-007 |

### 2.3 性能监控接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/{id}/monitor` | GET | 获取算法监控数据 | `sys:algorithm:monitor` | F-M05-008 |
| `/api/v1/algorithms/{id}/monitor/stats` | GET | 获取统计报表 | `sys:algorithm:monitor` | F-M05-008 |

### 2.4 模型预测接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/prediction` | POST | 执行模型预测 | - | F-M05-009 |
| `/api/v1/prediction/{taskId}` | GET | 查询预测任务状态 | - | F-M05-009 |
| `/api/v1/prediction/logs` | GET | 获取预测日志列表 | - | F-M05-009 |

### 2.5 效果评估接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/evaluation` | POST | 执行效果评估（PSNR/SSIM/LPIPS等） | - | F-M05-010 |
| `/api/v1/evaluation/{taskId}` | GET | 查询评估任务状态 | - | F-M05-010 |
| `/api/v1/evaluation/logs` | GET | 获取评估日志列表 | - | F-M05-010 |

## 3. 权限标识汇总

| 权限标识 | 说明 | 控制范围 |
|---------|------|---------|
| `sys:algorithm:add` | 新增算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:edit` | 编辑算法/修改状态 | 按钮显示 + 接口校验 |
| `sys:algorithm:audit` | 审核算法（通过/驳回） | 按钮显示 + 接口校验 |
| `sys:algorithm:stop` | 停用/启用算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:delete` | 删除算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:view` | 查看算法 | 默认所有用户 |
| `sys:algorithm:version` | 版本管理 | 按钮显示 + 接口校验 |
| `sys:algorithm:import` | 导入算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:export` | 导出算法 | 按钮显示 + 接口校验 |
| `sys:algorithm:monitor` | 性能监控 | 按钮显示 + 接口校验 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `B0001` | 业务错误 | 通用业务错误 |
| `B0200` | 算法不存在 | 查询/编辑/删除不存在的算法 |
| `B0201` | 算法名称已存在 | 新增/编辑/导入时名称重复 |
| `B0202` | 算法状态不允许该操作 | 当前状态下不能执行该操作 |
| `B0203` | 算法正在使用中,不能删除 | 算法仍有使用记录 |
| `B0204` | 模型文件不存在或已损坏 | 模型文件验证失败 |
| `B0205` | 版本号已存在 | 新增版本时版本号重复 |
| `B0206` | 当前版本不允许回滚 | 回滚到当前版本或已归档版本 |
| `B0207` | 审核权限不足 | 无审核权限操作审核接口 |
| `B0208` | 驳回原因不能为空 | 审核驳回时未填写原因 |
| `B0209` | 导入包格式错误 | 上传的不是有效的算法导入包 |
| `B0210` | 预测任务不存在 | 查询不存在的预测任务 |
| `B0211` | 预测任务已过期 | 查询超过有效期的任务 |
| `B0212` | 图片格式不支持 | 上传不支持的图片格式 |
| `B0220` | 评估任务不存在 | 查询不存在的评估任务 |
| `B0221` | 缺少清晰图进行对比 | 评估时缺少参考图像 |
| `A0230` | token无效或已过期 | 未认证访问 |

## 5. 接口详情查询

> 接口的详细请求参数、响应结构、Schema 定义可通过以下方式获取:
>
> 1. **API 文档 MCP**: 调用 `read_project_oas_wht4eg` 获取 OpenAPI Spec
> 2. **Swagger UI**: 访问 `/swagger-ui/index.html`(开发环境)

---

**文档版本**: v1.1.0
**最后更新**: 2026-01-18
**维护者**: 技术文档团队
