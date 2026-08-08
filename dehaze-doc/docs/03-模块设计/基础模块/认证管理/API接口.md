# 认证管理模块 API 接口

## 1. 文档概述

本文档定义 **认证管理** 模块的 HTTP API 规范。

- **基础路径**：`/api/v1/auth`
- **认证方式**：Session Cookie（`X-Session-Id`），Web 端自动携带，移动端通过 `X-Session-Id` 请求头传递
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

## 2. 接口清单

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/auth/login` | POST | 用户登录（支持 deviceType 多端会话隔离） | - | F-AM-001 |
| `/api/v1/auth/register` | POST | 用户注册 | - | F-AM-009 |
| `/api/v1/auth/logout` | POST | 用户注销 | - | F-AM-002 |
| `/api/v1/auth/captcha` | GET | 获取验证码 | - | F-AM-003 |
| `/api/v1/auth/me` | GET | 获取当前用户权限信息 | - | F-AM-005 |
| `/api/v1/auth/api-keys` | POST | 创建 API Key | - | F-AM-006 |
| `/api/v1/auth/api-keys` | GET | 查询当前用户的 API Key 列表 | - | F-AM-007 |
| `/api/v1/auth/api-keys/{id}` | DELETE | 删除/吊销 API Key | - | F-AM-008 |
| `/api/v1/auth/login-logs` | GET | 登录日志查询（按用户/时间/IP/状态筛选） | `sys:auth:log:list` | F-AM-010 |
| `/api/v1/auth/login-logs/export` | GET | 登录日志导出 | `sys:auth:log:export` | F-AM-010 |
| `/api/v1/auth/sessions` | GET | 查询用户在线会话列表 | `sys:auth:session:list` | F-AM-011 |
| `/api/v1/auth/sessions/{sessionId}` | DELETE | 踢出指定会话 | `sys:auth:session:kick` | F-AM-011 |

> **说明**：无独立的刷新令牌接口。Session 模式下后端自动处理会话续期，前端无需主动刷新令牌。

### 2.1 登录请求扩展字段

登录接口请求体新增可选字段 `deviceType`（设备类型，默认 `web`），取值：`web`/`android`/`flutter`/`miniprogram`，用于多端会话隔离。同一账号多端登录不互踢，Session 按 deviceType 区分存储。

## 3. 访问控制

| 接口 | 访问控制 |
|------|---------|
| 登录、注册、获取验证码 | 公开访问 |
| 注销、获取权限信息 | 需登录态 |
| 创建、查询、删除 API Key | 需登录态 |
| 登录日志查询/导出 | 需登录态；管理员查看全量，普通用户仅查看本人 |
| 在线会话查询、踢出 | 需登录态 + 对应权限标识（管理员） |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0210` | 用户名或密码错误 | 登录认证失败 |
| `A0211` | 账户已被禁用 | 禁用用户尝试登录 |
| `A0212` | 账户已被锁定 | 连续 5 次登录失败锁定 30 分钟 |
| `A0230` | 会话无效或已过期 | Session 不存在于 Redis |
| `A0213` | 验证码已过期 | 验证码超时 |
| `A0214` | 验证码错误 | 验证码校验失败 |
| `A0501` | 数据已存在 | 用户名已被注册 |
| `A0002` | 请求过于频繁 | 注册 IP 触发限流 |
