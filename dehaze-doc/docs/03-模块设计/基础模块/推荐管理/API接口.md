# 推荐管理模块 API 接口

## 1. 文档概述

本文档定义 **推荐管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/recommendations`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

> **重要**：接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档仅定义接口清单和权限标识。

## 2. 接口清单

### 2.1 推荐查询接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/recommendations/analyze` | POST | 图像特征分析 | - | F-REC-001 |
| `/api/v1/recommendations/algorithms` | GET | 获取算法推荐 | - | F-REC-002 |

### 2.2 反馈接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/recommendations/feedback` | POST | 提交推荐反馈 | - | F-REC-003 |

### 2.3 规则管理接口（管理员）

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/recommendations/rules` | GET | 获取推荐规则配置 | `sys:recommendation:rule:view` | F-REC-004 |
| `/api/v1/recommendations/rules` | PUT | 新增/更新推荐规则配置 | `sys:recommendation:rule:edit` | F-REC-004 |

> PUT `/rules` 契约：`id` 通过 query 参数传递（`id=0` 表示新增），请求体为规则表单（ruleName/sceneType/algorithmIds/weight/enabled）。同场景（sceneType）下已存在相同算法集合（algorithmIds，顺序无关）的规则时拒绝保存（A0501），新增与更新均校验（更新时排除自身）。

### 2.4 效果报表接口（管理员）

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/recommendations/report` | GET | 推荐效果报表 | `sys:recommendation:report` | - |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `sys:recommendation:rule:view` | 查看推荐规则配置（管理员） |
| `sys:recommendation:rule:edit` | 修改推荐规则配置（管理员） |
| `sys:recommendation:report` | 查看推荐效果报表（管理员） |
| - | 推荐查询和反馈提交接口登录用户即可访问 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 反馈记录不存在或非本人记录；更新的规则不存在；imageId 方式不支持 |
| `A0400` | 用户请求参数错误 | imageUrl 与 imageId 均未提供；规则权重超出 0-100；场景类型不合法；算法列表为空；报表日期格式非法 |
| `A0501` | 数据已存在 | 同场景下已存在相同算法组合的规则 |
| `A0701` | 文件格式不支持 | imageUrl 扩展名非 jpg/jpeg/png/webp/bmp/tiff/tif |
| `A0301` | 访问未授权 | 非管理员访问规则管理/报表接口 |
