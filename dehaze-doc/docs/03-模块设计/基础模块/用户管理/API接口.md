# 用户管理模块 API 接口

## 1. 文档概述

本文档定义 **用户管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/users`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。

## 2. 接口清单

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/users/page` | GET | 用户分页列表 | - | F-UM-001 |
| `/api/v1/users` | POST | 新增用户 | `sys:user:add` | F-UM-002 |
| `/api/v1/users/{userId}/form` | GET | 获取用户表单数据 | - | F-UM-003 |
| `/api/v1/users/{userId}` | PUT | 修改用户 | `sys:user:edit` | F-UM-003 |
| `/api/v1/users/{ids}` | DELETE | 删除用户（支持批量） | `sys:user:delete` | F-UM-004 |
| `/api/v1/users/{userId}/password` | PATCH | 重置用户密码 | `sys:user:password:reset` | F-UM-005 |
| `/api/v1/users/{userId}/status` | PATCH | 修改用户状态 | - | F-UM-006 |

> **导入导出接口**：用户模块的导出（`GET/POST /api/v1/users/_export`）、导入（`POST /api/v1/users/_import`）、模板下载（`GET /api/v1/users/template`）由通用导入导出框架 `GenericImportExportController` 统一实现，接口规范参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md) §8.2 通用CRUD接口模板，模块特定逻辑由 `UserExportHandler`/`UserImportHandler` 实现，详见 [后端实现.md](./后端实现.md)。

## 3. 权限标识汇总

> **完整权限矩阵**：详见 [需求规格.md](./需求规格.md) 第 2.2 节

| 权限标识 | 说明 |
|---------|------|
| `sys:user:add` | 新增用户 |
| `sys:user:edit` | 编辑用户 |
| `sys:user:delete` | 删除用户 |
| `sys:user:password:reset` | 重置密码 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0111` | 用户名已存在 | 新增/导入时用户名重复 |
| `A0200` | 用户不存在 | 编辑/删除不存在的用户 |
| `A0201` | 用户已禁用 | 禁用用户尝试登录 |
| `A0230` | 超级管理员不可删除 | 尝试删除超级管理员 |
| `A0231` | 超级管理员不可禁用 | 尝试禁用超级管理员 |

## 5. 接口详情查询

> 接口的详细请求参数、响应结构、Schema 定义可通过以下方式获取：
> 
> 1. **API 文档 MCP**：调用 `read_project_oas_yfcdew` 获取 OpenAPI Spec
> 2. **Swagger UI**：访问 `/swagger-ui/index.html`（开发环境）
