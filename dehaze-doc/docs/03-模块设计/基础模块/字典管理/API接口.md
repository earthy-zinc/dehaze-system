# 字典管理模块 API 接口

## 1. 文档概述

本文档定义 **字典管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/dict`（字典数据）、`/api/v1/dict/types`（字典类型）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。

## 2. 接口清单

### 2.1 字典类型接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/dict/types/page` | GET | 字典类型分页列表 | - | F-DM-001 |
| `/api/v1/dict/types` | POST | 新增字典类型 | `sys:dict:type:add` | F-DM-002 |
| `/api/v1/dict/types/{id}/form` | GET | 获取字典类型表单数据 | - | F-DM-003 |
| `/api/v1/dict/types/{id}` | PUT | 修改字典类型 | `sys:dict:type:edit` | F-DM-003 |
| `/api/v1/dict/types/{ids}` | DELETE | 删除字典类型（支持批量） | `sys:dict:type:delete` | F-DM-004 |

### 2.2 字典数据接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/dict/page` | GET | 字典分页列表 | - | F-DM-005 |
| `/api/v1/dict` | POST | 新增字典 | `sys:dict:data:add` | F-DM-006 |
| `/api/v1/dict/{id}/form` | GET | 获取字典数据表单数据 | - | F-DM-007 |
| `/api/v1/dict/{id}` | PUT | 修改字典 | `sys:dict:data:edit` | F-DM-007 |
| `/api/v1/dict/{ids}` | DELETE | 删除字典（支持批量） | `sys:dict:data:delete` | F-DM-008 |
| `/api/v1/dict/{typeCode}/options` | GET | 字典下拉选项 | - | F-DM-009 |

> **导入导出接口**：字典模块的导出（`GET/POST /api/v1/dict/_export`）、导入（`POST /api/v1/dict/_import`）、模板下载（`GET /api/v1/dict/template`）由通用导入导出框架统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §7，模块特定逻辑详见 [后端实现.md](./后端实现.md)。

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `sys:dict:type:add` | 新增字典类型 |
| `sys:dict:type:edit` | 编辑字典类型 |
| `sys:dict:type:delete` | 删除字典类型 |
| `sys:dict:data:add` | 新增字典数据 |
| `sys:dict:data:edit` | 编辑字典数据 |
| `sys:dict:data:delete` | 删除字典数据 |
| - | 字典查询接口登录用户即可访问 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0501` | 字典类型编码已存在 | 新增/编辑时编码重复 |
| `A0501` | 字典值已存在 | 同一类型下字典值重复 |
| `A0504` | 存在关联数据，无法删除 | 删除字典类型时存在关联字典数据 |


