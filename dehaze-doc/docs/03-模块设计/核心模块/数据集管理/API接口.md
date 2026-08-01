# 数据集管理模块 API 接口

## 1. 文档概述

本文档定义 **数据集管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：
  - 数据集：`/api/v1/datasets`
  - 数据项：`/api/v1/dataset-items`
  - 图片文件：`/api/v1/item-files`
  - 任务管理：`/api/v1/tasks`（统一任务接口）
- **公共约定**：参见 [../../02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)
- **后端实现**：[后端实现.md](./后端实现.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。

## 2. 接口清单

### 2.1 数据集接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/datasets` | GET | 获取数据集列表（支持树形） | - | F-M06-001 |
| `/api/v1/datasets/{id}` | GET | 获取数据集详情 | - | F-M06-003 |
| `/api/v1/datasets` | POST | 新增数据集 | `sys:dataset:add` | F-M06-002 |
| `/api/v1/datasets/{id}` | PUT | 修改数据集 | `sys:dataset:edit` | F-M06-002 |
| `/api/v1/datasets/{id}` | DELETE | 删除单个数据集 | `sys:dataset:delete` | F-M06-002 |
| `/api/v1/datasets/batch` | DELETE | 批量删除数据集 | `sys:dataset:delete` | F-M06-002 |
| `/api/v1/datasets/options` | GET | 获取数据集下拉选项 | - | F-M06-001 |

> **注意**：数据集导出功能已迁移至统一任务接口 `/api/v1/tasks`，请参见 [2.4 任务接口](#24-任务接口)

### 2.2 数据项接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/dataset-items` | GET | 分页查询数据项列表 | - | F-M06-004 |
| `/api/v1/dataset-items/{id}` | GET | 获取数据项详情 | - | F-M06-005 |
| `/api/v1/dataset-items` | POST | 创建空数据项 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/dataset-items/upload` | POST | 创建数据项并上传配对图片 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/dataset-items/batch` | POST | 批量创建数据项并上传图片 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/dataset-items/{id}` | PUT | 修改数据项信息 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/dataset-items/{id}` | DELETE | 删除数据项 | `sys:dataset:delete` | F-M06-008 |
| `/api/v1/dataset-items/batch` | DELETE | 批量删除数据项 | `sys:dataset:delete` | F-M06-008 |

### 2.3 图片文件接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/item-files/{id}` | GET | 获取图片详细信息 | - | F-M06-005 |
| `/api/v1/item-files` | POST | 上传数据项图片 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/item-files/{id}` | PUT | 修改图片信息 | `sys:dataset:edit` | F-M06-008 |
| `/api/v1/item-files/{id}` | DELETE | 删除图片 | `sys:dataset:delete` | F-M06-008 |
| `/api/v1/item-files/batch` | DELETE | 批量删除图片 | `sys:dataset:delete` | F-M06-008 |

### 2.4 任务接口（统一）

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/tasks` | POST | 创建任务（支持多种类型） | 按任务类型区分 | F-M06-002 |
| `/api/v1/tasks/{taskId}` | GET | 查询任务状态 | - | F-M06-002 |
| `/api/v1/tasks/{taskId}/download` | GET | 下载任务结果文件 | - | F-M06-002 |
| `/api/v1/tasks/{taskId}` | DELETE | 取消任务 | - | F-M06-002 |
| `/api/v1/tasks` | GET | 分页查询任务列表 | - | F-M06-002 |

**任务类型说明**：

| 任务类型 | type 值 | 说明 | 权限标识 |
|---------|---------|------|---------|
| 数据集导出 | `dataset_export` | 导出整个数据集为 ZIP | `sys:dataset:export` |
| 数据项下载 | `item_download` | 下载指定数据项的图片 | - |
| 批量下载 | `batch_download` | 批量下载多个图片 | - |

## 3. 权限标识汇总

| 权限标识 | 说明 | 控制范围 |
|---------|------|---------|
| `sys:dataset:add` | 新增数据集 | 按钮显示 + 接口校验 |
| `sys:dataset:edit` | 编辑数据集/数据项/图片 | 按钮显示 + 接口校验 |
| `sys:dataset:delete` | 删除数据集/数据项/图片 | 按钮显示 + 接口校验 |
| `sys:dataset:view` | 查看数据集 | 默认所有用户 |
| `sys:dataset:export` | 导出数据集 | 按钮显示 + 接口校验 |

## 4. 业务错误码

数据集模块复用系统通用错误码，不定义模块专属错误码：

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0400` | 参数错误 | 必填字段缺失、名称同级重复、配对图片分辨率不一致、层级超限、上传非图片格式等 |
| `A0401` | 资源不存在 | 查询/编辑/删除不存在的数据集/数据项/图片 |
| `A0500` | 业务异常 | 通用业务异常（兜底） |
| `A0230` | token无效或已过期 | 未认证访问 |

> 数据集导出/下载任务相关错误码由任务管理模块统一管理，详见 [任务管理/API接口.md](../../基础模块/任务管理/API接口.md)

## 5. 业务错误码
