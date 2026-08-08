# AI 图像处理平台 - API 规范

## 1. 文档概述

### 1.1 文档目的

本文档定义 AI 图像处理平台的全局 HTTP API 规范，包括请求/响应格式、状态码、认证方式、分页约定等，作为项目级 API 契约的唯一权威来源。

### 1.2 适用范围

本规范适用于系统所有 RESTful API 接口设计与实现，前后端开发人员、测试人员须严格遵循。

### 1.3 设计原则

| 原则 | 说明 |
|-----|------|
| **RESTful 风格** | 资源导向设计，使用标准 HTTP 方法（GET/POST/PUT/PATCH/DELETE） |
| **统一响应格式** | 所有接口返回统一的 JSON 结构，便于前端统一处理 |
| **语义化状态码** | 采用业务状态码体系，分类清晰、便于定位问题 |
| **版本化管理** | API 路径包含版本号（如 `/api/v1`），支持平滑升级 |

### 1.4 相关文档

- 总体架构设计：`02-系统架构/01-总体架构设计.md`
- 环境与兼容性要求：`02-系统架构/02-环境与兼容性要求.md`

---

## 2. 公共约定

### 2.1 基础路径

| 环境 | 基础路径 |
|-----|---------|
| 开发环境 | `http://localhost:8989/api/v1` |
| 生产环境 | `http://<server-ip>:8989/api/v1` |

### 2.2 请求头

| 请求头 | 是否必填 | 说明 | 示例 |
|--------|---------|------|------|
| `X-Session-Id` | 是（鉴权接口） | Session ID | `a1b2c3d4...`（Web 端通过 Cookie 自动携带） |
| `Authorization` | 是（API Key 接口） | API Key 凭证 | `Bearer dhak_a1b2c3d4...` |
| `Content-Type` | 是（POST/PUT/PATCH） | 请求内容类型 | `application/json` |
| `Accept` | 否 | 期望响应格式 | `application/json` |

### 2.3 HTTP 方法语义

| 方法 | 语义 | 幂等性 | 示例场景 |
|------|------|--------|---------|
| `GET` | 查询资源 | 是 | 获取用户列表、详情 |
| `POST` | 创建资源 | 否 | 新增用户、上传文件 |
| `PUT` | 全量更新 | 是 | 修改用户信息 |
| `PATCH` | 部分更新 | 是 | 修改用户状态、密码 |
| `DELETE` | 删除资源 | 是 | 删除用户、批量删除 |

### 2.4 URL 命名规范

| 规则 | 正确示例 | 错误示例 |
|------|---------|---------|
| 使用 kebab-case | `/api/v1/dataset-items` | `/api/v1/datasetItems` |
| 资源名用复数 | `/api/v1/users` | `/api/v1/user` |
| 路径参数用 `{id}` | `/api/v1/users/{userId}` | `/api/v1/users/:userId` |
| 操作用下划线前缀 | `/api/v1/users/_export` | `/api/v1/users/export` |

---

## 3. 统一响应格式

### 3.1 基础响应结构

所有接口响应统一使用以下 JSON 结构：

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {},
  "traceId": "abc123def456"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | string | 是 | 业务状态码，成功为 `"00000"` |
| `msg` | string | 是 | 状态描述信息，错误时包含具体错误原因 |
| `data` | object/array/null | 是 | 业务数据，无数据时为 `null` |
| `traceId` | string | 否 | 请求追踪 ID，用于问题排查 |

### 3.2 错误响应说明

接口出错时无独立的 `errors` 字段，错误信息统一通过 `msg` 返回。参数校验失败时，多个字段的错误信息以 `；` 拼接后放入 `msg`：

```json
{
  "code": "A0400",
  "msg": "用户名不能为空；邮箱格式不正确",
  "data": null,
  "traceId": "abc123def456"
}
```

---

## 4. 分页约定

### 4.1 分页请求参数

| 参数 | 类型 | 必填 | 说明 | 默认值 | 约束 |
|------|------|------|------|--------|------|
| `pageNum` | integer | 否 | 页码（从 1 开始） | 1 | ≥ 1 |
| `pageSize` | integer | 否 | 每页条数 | 10 | 1 ~ 100 |

### 4.2 分页请求示例

```http
GET /api/v1/users/page?pageNum=1&pageSize=10&keywords=admin HTTP/1.1
Cookie: X-Session-Id=<sessionId>
```

### 4.3 分页响应示例

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "list": [
      {
        "id": 1,
        "username": "admin",
        "nickname": "管理员",
        "status": 1,
        "createTime": "2024-01-01 00:00:00"
      }
    ],
    "total": 50
  },
  "traceId": "abc123def456"
}
```

---

## 5. 状态码体系

### 5.1 状态码分类

系统采用 5 位字符串状态码，按首字母分类：

| 分类 | 范围 | 说明 |
|------|------|------|
| **成功** | `00000` | 操作成功 |
| **A 类** | `A0xxx` | 用户端错误（参数、认证、权限、文件上传等） |
| **B 类** | `B0xxx` | 系统端错误（执行、超时、资源） |
| **C 类** | `C0xxx` | 第三方服务错误（中间件、消息、数据库） |

### 5.2 HTTP 状态码映射

| HTTP Status | 适用场景 | 对应业务码 |
|-------------|---------|-----------|
| `200 OK` | 请求成功 | `00000` |
| `400 Bad Request` | 参数错误 | `A04xx` |
| `401 Unauthorized` | 未认证 | `A02xx`, `A0301` |
| `403 Forbidden` | 无权限 | `A03xx` |
| `404 Not Found` | 资源不存在 | `A0401` |
| `500 Internal Server Error` | 服务器错误 | `B0xxx`, `C0xxx` |

---

## 6. 时间格式

### 6.1 约定

| 场景 | 格式 | 示例 |
|------|------|------|
| 请求参数（日期时间） | `yyyy-MM-dd HH:mm:ss` | `2024-01-01 00:00:00` |
| 请求参数（日期） | `yyyy-MM-dd` | `2024-01-01` |
| 请求参数（时间戳） | Unix 毫秒时间戳 | `1704067200000` |
| 响应数据 | `yyyy-MM-dd HH:mm:ss` | `2024-01-01 00:00:00` |

### 6.2 时区

- 服务端统一使用 **Asia/Shanghai (UTC+8)** 时区
- 前端展示时根据用户时区进行转换

---

## 7. 通用 CRUD 接口模板

系统 API 按业务模块划分。

| 路径 | 方法 | 功能 | 权限标识 |
|------|------|------|---------|
| `/{module}/page` | GET | 分页列表 | - |
| `/{module}` | GET | 列表（不分页） | - |
| `/{module}` | POST | 新增 | `{module}:add` |
| `/{module}/{id}` | GET | 详情 | - |
| `/{module}/{id}` | PUT | 修改 | `{module}:edit` |
| `/{module}/{id}` | DELETE | 删除 | `{module}:delete` |
| `/{module}/{ids}` | DELETE | 批量删除 | `{module}:delete` |
| `/{module}/_export` | GET | 导出（简单查询条件） | `{module}:export` |
| `/{module}/_export` | POST | 导出（复杂查询条件） | `{module}:export` |
| `/{module}/_import` | POST | 导入 | `{module}:import` |
| `/{module}/template` | GET | 下载导入模板 | `{module}:import` |

## 8. 错误处理规范

系统统一使用全局异常处理器（Java / Go / Python 三端均实现），所有异常均返回第 3 节定义的标准响应格式，错误信息通过 `msg` 字段表达，HTTP 状态码用于区分错误大类（见 §5.2）。

前端仅需根据 `code` 是否等于 `00000` 判断请求是否成功，失败时直接展示 `msg`，无需针对具体错误码编写分支逻辑。

---

## 9. 版本管理

### 9.1 版本策略

- 当前版本：`v1`
- 版本号包含在 URL 路径中：`/api/v1/xxx`
- 重大变更发布新版本（如 `v2`），旧版本保留兼容期

### 9.2 OpenAPI 对外开放

系统对外提供 OpenAPI 3.0 规范文档，支持第三方开发者集成：

- **文档生成**：由后端代码注解（Java: SpringDoc / Go: Swag / Python: FastAPI）自动生成 OpenAPI 规范，无需手动维护
- **定义来源**：以 Java 端 Controller 接口定义为 OpenAPI spec 生成来源（Spring Boot 原生支持 OpenAPI 3 文档）。Java/Go/Python 三端后端 API 完全相同，Java 端作为业务主端是 spec 的唯一权威来源
- **访问地址**：`/v3/api-docs`（JSON 格式）、`/swagger-ui.html`（Swagger UI 交互式文档）
- **API Key 鉴权**：外部调用通过 `Authorization: Bearer dhak_xxx` 进行认证，详见 [认证管理/API Key认证.md](../03-模块设计/基础模块/认证管理/API Key认证.md)
- **多端 SDK 自动生成**：使用 `openapi-generator` 按 OpenAPI spec 自动生成多端 SDK（TypeScript/Java/Dart/Kotlin 等），生成的 SDK 作为接口契约层
- **与手写 SDK 的关系**：现有手写 SDK（`dehaze-sdk-js`、`dehaze-android/sdk`）的网络层封装（拦截器、错误处理、trace_id 透传）保留为上层封装，生成层替换手写的接口定义部分
