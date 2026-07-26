# 角色管理模块 - API接口

## 1. 文档概述

本文档定义 **角色管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/roles`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

**基础路径**：`/api/v1/roles`

| 接口路径 | 方法 | 功能描述 | 权限标识 |
|---------|------|---------|---------|
| `/api/v1/roles/page` | GET | 角色分页列表 | - |
| `/api/v1/roles/options` | GET | 角色下拉选项 | - |
| `/api/v1/roles` | POST | 新增角色 | sys:role:add |
| `/api/v1/roles/{roleId}/form` | GET | 获取角色表单数据 | - |
| `/api/v1/roles/{id}` | PUT | 修改角色 | sys:role:edit |
| `/api/v1/roles/{ids}` | DELETE | 删除角色 | sys:role:delete |
| `/api/v1/roles/{roleId}/status` | PUT | 修改角色状态 | sys:role:edit |
| `/api/v1/roles/{roleId}/menuIds` | GET | 获取角色菜单 ID 集合 | - |
| `/api/v1/roles/{roleId}/menus` | PUT | 分配菜单权限 | sys:role:edit |

> **导入导出接口**：角色模块的导出（`GET/POST /api/v1/roles/_export`）、导入（`POST /api/v1/roles/_import`）、模板下载（`GET /api/v1/roles/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §8.2 通用CRUD接口模板，模块特定逻辑由 `RoleExportHandler`/`RoleImportHandler` 实现，详见 [后端实现.md](./后端实现.md)。

## 3. 权限标识汇总

| 权限标识 | 功能 | 控制方式 |
|---------|------|---------|
| sys:role:add | 新增角色 | 按钮显示 + 接口校验 |
| sys:role:edit | 编辑角色 | 按钮显示 + 接口校验 |
| sys:role:delete | 删除角色 | 按钮显示 + 接口校验 |

## 4. 数据权限选项

| 权限值 | 名称 | 说明 |
|-------|------|------|
| 0 | 全部数据 | 可访问系统所有数据 |
| 1 | 部门及子部门数据 | 可访问本部门及下级部门数据 |
| 2 | 本部门数据 | 仅可访问本部门数据（默认） |
| 3 | 本人数据 | 仅可访问自己的数据 |

## 5. 状态枚举

| 状态值 | 显示 | 说明 |
|--------|------|------|
| 1 | 启用（绿色标签） | 用户可以使用该角色权限 |
| 0 | 禁用（灰色标签） | 用户无法使用该角色权限 |

## 6. 业务错误码

| 错误码 | 错误信息 | 触发场景 |
|-------|---------|---------|
| ROLE_CODE_EXISTS | 角色编码已存在 | 创建时角色编码重复 |
| ROLE_NAME_EXISTS | 角色名称已存在 | 创建时角色名称重复 |
| ROLE_NOT_FOUND | 角色不存在 | 编辑/删除时角色不存在 |
| ROLE_HAS_USERS | 该角色下已关联用户，无法删除 | 删除时角色已关联用户 |
| ROOT_ROLE_PROTECTED | 超级管理员角色不可删除 | 删除超级管理员角色 |

## 6. 接口详细文档

> **注意**：完整的接口参数/响应定义可通过 **API文档 MCP** 查询获得。