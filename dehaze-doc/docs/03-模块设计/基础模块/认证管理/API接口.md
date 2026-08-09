# 认证管理模块 API 接口

## 1. 文档概述

本文档定义 **认证管理** 模块的 HTTP API 规范。

- **基础路径**：`/api/v1/auth`
- **认证方式**：Session Cookie（`X-Session-Id`）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

## 2. 接口清单

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/auth/login` | POST | 用户登录 | - | F-AM-001 |
| `/api/v1/auth/register` | POST | 用户注册 | - | F-AM-009 |
| `/api/v1/auth/logout` | POST | 用户注销 | - | F-AM-002 |
| `/api/v1/auth/captcha` | GET | 获取验证码 | - | F-AM-003 |
| `/api/v1/auth/me` | GET | 获取当前用户权限信息 | - | F-AM-005 |
| `/api/v1/auth/api-keys` | POST | 创建 API Key | - | F-AM-006 |
| `/api/v1/auth/api-keys` | GET | 查询当前用户的 API Key 列表 | - | F-AM-007 |
| `/api/v1/auth/api-keys/{id}` | DELETE | 删除/吊销 API Key | - | F-AM-008 |
| `/api/v1/auth/login-logs` | GET | 登录日志查询 | `sys:auth:log:list` | F-AM-010 |
| `/api/v1/auth/login-logs/export` | GET | 登录日志导出 | `sys:auth:log:export` | F-AM-010 |
| `/api/v1/auth/sessions` | GET | 查询用户在线会话列表 | `sys:auth:session:list` | F-AM-011 |
| `/api/v1/auth/sessions/{sessionId}` | DELETE | 踢出指定会话 | `sys:auth:session:kick` | F-AM-011 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| `sys:auth:log:list` | 登录日志查询 |
| `sys:auth:log:export` | 登录日志导出 |
| `sys:auth:session:list` | 在线会话查询 |
| `sys:auth:session:kick` | 踢出指定会话 |

> **权限标识说明**：接口清单中 `-` 表示无需特定权限标识。登录、注册、获取验证码为公开访问；注销、获取权限信息、API Key 管理为登录用户可访问；登录日志查询/导出需登录态，管理员查看全量、普通用户仅查看本人。

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0002` | 请求过于频繁 | 注册 IP 触发限流 |
| `A0210` | 用户名或密码错误 | 登录认证失败 |
| `A0211` | 账户已被禁用 | 禁用用户尝试登录 |
| `A0212` | 账户已被锁定 | 连续 5 次登录失败锁定 30 分钟 |
| `A0213` | 验证码已过期 | 验证码超时 |
| `A0214` | 验证码错误 | 验证码校验失败 |
| `A0230` | 会话无效或已过期 | Session 失效或过期 |
| `A0501` | 数据已存在 | 用户名已被注册 |
