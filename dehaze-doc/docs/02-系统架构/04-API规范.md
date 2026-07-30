# 图像去雾系统 - API 规范

## 1. 文档概述

### 1.1 文档目的

本文档定义图像去雾系统的全局 HTTP API 规范，包括请求/响应格式、状态码、认证机制、分页约定等，作为项目级 API 契约的唯一权威来源。

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
- API 接口详情：通过 OpenAPI MCP 工具获取（`read_project_oas_yfcdew`）

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
  "traceId": "abc123def456",
  "timestamp": 1737100800000,
  "errors": []
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | string | 是 | 业务状态码，成功为 `"00000"` |
| `msg` | string | 是 | 状态描述信息 |
| `data` | object/array/null | 是 | 业务数据，无数据时为 `null` |
| `traceId` | string | 否 | 请求追踪 ID，用于问题排查 |
| `timestamp` | number | 否 | 响应时间戳（毫秒） |
| `errors` | array | 否 | 错误详情列表（参数校验失败时） |

### 3.2 错误详情结构

当请求参数校验失败时，`errors` 字段包含详细错误信息：

```json
{
  "code": "A0400",
  "msg": "用户请求参数错误",
  "data": null,
  "errors": [
    {
      "field": "username",
      "message": "用户名不能为空",
      "code": "NotBlank"
    },
    {
      "field": "email",
      "message": "邮箱格式不正确",
      "code": "Email"
    }
  ]
}
```

### 3.3 分页响应结构

分页接口统一使用以下响应格式：

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "list": [],
    "total": 100
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `data.list` | array | 当前页数据列表 |
| `data.total` | number | 总记录数 |

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
  }
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

### 5.2 完整状态码清单

#### 成功状态码

| code | msg | 说明 |
|------|-----|------|
| `00000` | 一切ok | 操作成功 |

#### A 类：用户端错误

| code | msg | 说明 |
|------|-----|------|
| `A0001` | 用户端错误 | 通用用户端错误 |
| `A0002` | 您的请求已提交，请不要重复提交或等待片刻再尝试。 | 重复提交 |

**认证相关 (A02xx)**

| code | msg | 说明 |
|------|-----|------|
| `A0200` | 用户登录异常 | 登录异常通用 |
| `A0201` | 用户不存在 | 用户账号不存在 |
| `A0202` | 用户账户被冻结 | 账户被禁用 |
| `A0203` | 用户账户已作废 | 账户已注销 |
| `A0210` | 用户名或密码错误 | 凭证错误 |
| `A0211` | 用户输入密码次数超限 | 密码尝试锁定 |
| `A0212` | 客户端认证失败 | OAuth 客户端认证失败 |
| `A0213` | 验证码已过期 | 验证码超时 |
| `A0214` | 验证码错误 | 验证码不匹配 |
| `A0230` | token无效或已过期 | Session 不存在或已过期 |
| `A0231` | token已被禁止访问 | Session 已失效或已注销 |

**权限相关 (A03xx)**

| code | msg | 说明 |
|------|-----|------|
| `A0300` | 访问权限异常 | 权限异常通用 |
| `A0301` | 访问未授权 | 未登录或无权限 |
| `A0302` | 演示环境禁止新增、修改和删除数据，请本地部署后测试 | 演示环境限制 |

**参数相关 (A04xx)**

| code | msg | 说明 |
|------|-----|------|
| `A0400` | 用户请求参数错误 | 参数校验失败 |
| `A0401` | 请求资源不存在 | 资源 404 |
| `A0410` | 请求必填参数为空 | 必填参数缺失 |

**业务规则 (A05xx)**

|| code | msg | 说明 |
||------|-----|------|
|| `A0500` | 业务异常 | 业务逻辑异常 |
|| `A0501` | 数据已存在 | 唯一性约束冲突 |
|| `A0502` | 数据状态不允许 | 当前状态不允许此操作 |
|| `A0503` | 操作不允许 | 不满足操作前置条件 |

**操作相关 (A06xx)**

|| code | msg | 说明 |
||------|-----|------|
|| `A0600` | 操作失败 | 操作执行失败 |
|| `A0601` | 操作已完成 | 操作已完成，请勿重复 |

**文件上传与导入导出 (A07xx)**

| code | msg | 说明 |
|------|-----|------|
| `A0700` | 用户上传文件异常 | 文件上传通用错误 |
| `A0701` | 文件格式不支持 | 上传非 Excel/CSV 文件 |
| `A0702` | 文件大小超限 | 上传文件 > 20MB |
| `A0703` | 文件内容为空 | 上传空文件或无数据行 |
| `A0704` | 文件解析失败 | Excel/CSV 格式错误 |
| `A0705` | 模板字段不匹配 | 导入文件表头与模板不一致 |
| `A0706` | 必填字段为空 | 导入数据缺少必填字段 |
| `A0707` | 数据校验失败 | 字段格式/唯一性校验不通过 |
| `A0708` | 导入数据超出限制 | 单次导入超过 10 万行 |
| `A0709` | 导出行数超出限制 | 单次导出超过 10 万行 |
| `A0710` | 不支持该模块导入 | 数据集等不支持导入的模块 |

#### B 类：系统端错误

| code | msg | 说明 |
|------|-----|------|
| `B0001` | 系统执行出错 | 通用系统错误 |
| `B0100` | 系统执行超时 | 执行超时 |
| `B0101` | 系统订单处理超时 | 业务处理超时 |

**容灾与限流 (B02xx)**

| code | msg | 说明 |
|------|-----|------|
| `B0200` | 系统容灾功能被触发 | 容灾降级 |
| `B0210` | 系统并发限流 | 并发限流保护 |
| `B0211` | 系统速率限流 | 速率限制保护 |
| `B0220` | 系统功能降级 | 服务降级 |

**资源相关 (B03xx)**

| code | msg | 说明 |
|------|-----|------|
| `B0300` | 系统资源异常 | 资源异常通用 |
| `B0308` | 导出任务并发超限 | 单用户导入导出任务并发数超限 |
| `B0310` | 系统资源耗尽 | 资源不足 |
| `B0320` | 系统资源访问异常 | 资源访问失败 |
| `B0321` | 系统读取磁盘文件失败 | 磁盘读取失败 |

#### C 类：第三方服务错误

| code | msg | 说明 |
|------|-----|------|
| `C0001` | 调用第三方服务出错 | 第三方调用通用 |
| `C0100` | 中间件服务出错 | 中间件错误 |
| `C0113` | 接口不存在 | 接口未定义 |

**消息服务 (C012x)**

| code | msg | 说明 |
|------|-----|------|
| `C0120` | 消息服务出错 | 消息服务通用 |
| `C0121` | 消息投递出错 | 消息发送失败 |
| `C0122` | 消息消费出错 | 消息消费失败 |
| `C0123` | 消息订阅出错 | 订阅失败 |
| `C0124` | 消息分组未查到 | 消费组不存在 |

**缓存服务 (C02xx)**

|| code | msg | 说明 |
||------|-----|------|
|| `C0200` | 缓存服务出错 | 缓存服务通用 |
|| `C0201` | 缓存未命中 | 缓存中不存在 |
|| `C0202` | 缓存写入失败 | 数据写入缓存失败 |

**对象存储 (C04xx)**

|| code | msg | 说明 |
||------|-----|------|
|| `C0400` | 对象存储服务出错 | 对象存储通用 |
|| `C0401` | 文件上传失败 | 上传文件失败 |
|| `C0402` | 文件下载失败 | 下载文件失败 |

**数据库服务 (C03xx)**

| code | msg | 说明 |
|------|-----|------|
| `C0300` | 数据库服务出错 | 数据库通用错误 |
| `C0311` | 表不存在 | 数据表缺失 |
| `C0312` | 列不存在 | 字段缺失 |
| `C0321` | 多表关联中存在多个相同名称的列 | 字段名冲突 |
| `C0331` | 数据库死锁 | 死锁异常 |
| `C0341` | 主键冲突 | 唯一键冲突 |

### 5.3 HTTP 状态码映射

| HTTP Status | 适用场景 | 对应业务码 |
|-------------|---------|-----------|
| `200 OK` | 请求成功 | `00000` |
| `400 Bad Request` | 参数错误 | `A04xx` |
| `401 Unauthorized` | 未认证 | `A02xx`, `A0301` |
| `403 Forbidden` | 无权限 | `A03xx` |
| `404 Not Found` | 资源不存在 | `A0401` |
| `500 Internal Server Error` | 服务器错误 | `B0xxx`, `C0xxx` |

---

## 6. 认证机制

### 6.1 Session 认证

系统采用 Session 进行身份认证，Session ID 通过 Cookie（`X-Session-Id`）或请求头传递，由 Redis 管理会话状态。

**认证流程：**

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Server as 服务端
    participant Redis as Redis

    Client->>Server: POST /api/v1/auth/login (username, password)
    Server->>Server: 校验凭证
    Server->>Redis: 存储 Session 信息
    Server-->>Client: { sessionId, user } (Set-Cookie: X-Session-Id)

    Client->>Server: GET /api/v1/auth/me (Cookie: X-Session-Id)
    Server->>Redis: 校验 Session 状态
    Server-->>Client: { code: "00000", data: {...} }
```

### 6.2 登录接口

**请求：**

```http
POST /api/v1/auth/login HTTP/1.1
Content-Type: application/json

{
  "username": "admin",
  "password": "Dehaze@2026",
  "captchaKey": "abc123",
  "captchaCode": "8v9a",
  "rememberMe": false
}
```

**响应：**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "sessionId": "a1b2c3d4...",
    "user": {
      "id": 1,
      "username": "admin",
      "nickname": "管理员"
    }
  }
}
```

> `sessionId` 通过 `Set-Cookie: X-Session-Id={sessionId}` 自动下发给 Web 端；移动端需从响应数据中提取并存储，后续请求通过 `X-Session-Id` 请求头传递。详见 [认证管理/API接口.md](../03-模块设计/基础模块/认证管理/API接口.md)。

### 6.3 Session 使用

所有需要认证的接口须通过 Cookie 或请求头携带 Session ID：

```http
GET /api/v1/auth/me HTTP/1.1
Cookie: X-Session-Id=a1b2c3d4...
```

### 6.4 退出登录

```http
DELETE /api/v1/auth/logout HTTP/1.1
Cookie: X-Session-Id=<sessionId>
```

### 6.5 API Key 认证

除 Session 认证外，系统支持 **API Key** 作为长期身份凭证，面向脚本调用、定时任务、第三方系统集成等机器对机器（M2M）场景。API Key 默认永不过期，可选设置过期时间。

**凭证格式：**

API Key 明文带固定前缀 `dhak_`，例如 `dhak_a1b2c3d4e5f6...`。

**携带方式：**

通过 `Authorization: Bearer` 请求头携带：

```http
GET /api/v1/datasets/page?pageNum=1&pageSize=10 HTTP/1.1
Authorization: Bearer dhak_a1b2c3d4e5f6...
```

服务端识别到 `dhak_` 前缀的凭证时走 API Key 校验分支，校验通过后以 Key 所属用户身份继续后续鉴权。

**管理接口：**

| 路径 | 方法 | 功能 |
|------|------|------|
| `/api/v1/auth/api-keys` | POST | 创建 API Key（请求体：`{"name": "xxx", "expiresAt": "可选 ISO 日期时间"}`） |
| `/api/v1/auth/api-keys` | GET | 查询当前用户的 API Key 列表 |
| `/api/v1/auth/api-keys/{id}` | DELETE | 删除/吊销 API Key |

**跨后端支持：**

API Key 存储于共享数据库，**Java / Go / Python** 三个后端服务通用，用户只需创建一个 Key 即可在任意后端发起的接口调用中使用。

**安全约定：**

| 安全措施 | 说明 |
|---------|------|
| 明文仅展示一次 | API Key 明文仅在创建成功时返回一次，之后无法再查询 |
| 哈希存储 | 服务端仅存储 API Key 的 SHA-256 哈希值，不存储明文 |
| 可吊销 | 用户可随时删除/吊销 Key，删除后立即失效 |
| 可选过期 | 支持创建时设置 `expiresAt`，到期后自动失效 |

> 详细设计与使用说明参见 `03-模块设计/基础模块/认证管理/API Key认证.md`。

---

## 7. 时间格式

### 7.1 约定

| 场景 | 格式 | 示例 |
|------|------|------|
| 请求参数（日期时间） | `yyyy-MM-dd HH:mm:ss` | `2024-01-01 00:00:00` |
| 请求参数（日期） | `yyyy-MM-dd` | `2024-01-01` |
| 请求参数（时间戳） | Unix 毫秒时间戳 | `1704067200000` |
| 响应数据 | `yyyy-MM-dd HH:mm:ss` | `2024-01-01 00:00:00` |

### 7.2 时区

- 服务端统一使用 **Asia/Shanghai (UTC+8)** 时区
- 前端展示时根据用户时区进行转换

---

## 8. 接口模块清单

系统 API 按业务模块划分，完整接口详情通过 OpenAPI MCP 工具获取。

### 8.1 模块总览

| 模块 | 路径前缀 | 说明 |
|------|---------|------|
| **认证中心** | `/api/v1/auth` | 登录、登出、验证码、API Key 管理 |
| **用户管理** | `/api/v1/users` | 用户 CRUD、导入导出 |
| **角色管理** | `/api/v1/roles` | 角色 CRUD、权限分配 |
| **菜单管理** | `/api/v1/menus` | 菜单 CRUD、路由配置 |
| **部门管理** | `/api/v1/depts` | 部门树形管理 |
| **字典管理** | `/api/v1/dict` | 字典类型与数据 |
| **文件管理** | `/api/v1/files` | 文件上传下载 |
| **数据集管理** | `/api/v1/datasets` | 数据集 CRUD |
| **数据项管理** | `/api/v1/dataset-items` | 数据项上传、配对 |
| **数据项图片** | `/api/v1/item-files` | 数据项图片管理 |
| **算法管理** | `/api/v1/algorithms` | 算法配置 |

### 8.2 通用 CRUD 接口模板

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

> **说明**：导入导出接口由通用框架 `GenericImportExportController` 统一实现，各模块只需实现 `ExportHandler`/`ImportHandler` 处理器，不各自定义 Controller。GET 导出用于简单查询条件（列表页筛选参数），POST 导出用于复杂查询条件（请求体传递），两者内部调用同一 Service 方法。

#### 8.2.1 导出接口参数

**GET/POST `/{module}/_export` 请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| format | string | 否 | 文件格式：`excel`（默认） / `csv` |
| async | boolean | 否 | 是否强制异步：`true` / `false` / 不传（自动判断，数据量>1000 条走异步） |
| fields | string | 否 | 导出字段，逗号分隔（不传则导出全部字段） |
| + 模块特定查询参数 | - | 否 | 各模块列表查询参数（如 keywords/status 等，导出忽略分页参数） |

**响应**：
- **同步模式**（数据量小）：直接返回文件流
  ```
  Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
  Content-Disposition: attachment; filename="users_20260726_153000.xlsx"
  Body: <文件二进制流>
  ```
- **异步模式**（数据量大）：返回 JSON
  ```json
  {
    "code": "00000",
    "msg": "导出任务已创建",
    "data": { "taskId": "...", "status": "PENDING", "estimatedCount": 50000 }
  }
  ```

#### 8.2.2 导入接口参数

**POST `/{module}/_import` 请求参数**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 上传的文件（Excel/CSV），≤20MB |
| mode | string | 否 | 导入模式：`all`（全量，默认） / `partial`（部分，跳过错误行） |
| async | boolean | 否 | 是否异步：`true` / `false` / 不传（数据量>1000 行自动异步） |
| + 模块特定参数 | - | 否 | 如用户导入的 `deptId`、角色导入的 `defaultDataScope` 等 |

**响应**：
- **同步模式**：返回导入结果
  ```json
  {
    "code": "00000",
    "msg": "导入完成",
    "data": {
      "totalRows": 100, "successCount": 95, "failureCount": 5, "skippedCount": 0,
      "errors": [{ "row": 3, "field": "username", "message": "用户名已存在" }],
      "errorReportUrl": null
    }
  }
  ```
- **异步模式**：返回 taskId
  ```json
  { "code": "00000", "msg": "导入任务已创建", "data": { "taskId": "...", "status": "PENDING" } }
  ```

#### 8.2.3 模板下载接口

**GET `/{module}/template?format=excel|csv`**：直接返回文件流（动态生成，包含表头和示例数据）。

#### 8.2.4 支持导入导出的模块

| 模块 | module 标识 | 导出 | 导入 | 备注 |
|------|------------|------|------|------|
| 用户管理 | `user` | ✅ | ✅ | - |
| 角色管理 | `role` | ✅ | ✅ | - |
| 部门管理 | `dept` | ✅ | ✅ | 树形导出为扁平结构 |
| 菜单管理 | `menu` | ✅ | ✅ | 树形导出为扁平结构 |
| 字典管理 | `dict` | ✅ | ✅ | 字典类型+字典数据 |
| 数据集管理 | `dataset` | ✅ | ❌ | 仅导出（ZIP 打包） |
| 算法管理 | `algorithm` | ✅ | ✅ | Excel/CSV 元数据，不含权重文件 |

### 8.3 预测/评估异步任务接口

预测与评估为计算密集型任务，统一采用**异步任务模式**：POST 立即返回 `logId + status="processing"`，前端通过 GET 轮询直到终态（`completed` / `failed`）。

#### 8.3.1 POST `/api/v1/prediction`

请求体不变（`PredictionForm`）。响应变更：

```json
{
  "code": "00000",
  "data": { "logId": 88, "status": "processing" }
}
```

`PredictionResultVO` 字段：

| 字段 | 类型 | 返回时机 |
|------|------|---------|
| `logId` | Long | POST + GET |
| `status` | String | POST + GET（`processing`/`completed`/`failed`） |
| `resultUrl` | String | GET `completed` 时 |
| `resultThumbnailUrl` | String | GET `completed` 时 |
| `time` | int | GET `completed`/`failed` 时 |
| `errorMessage` | String | GET `failed` 时 |

#### 8.3.2 GET `/api/v1/prediction/{taskId}`

根据 `status` 返回不同字段：`processing` 仅返回 `logId + status`；`completed` 返回完整结果；`failed` 返回 `errorMessage + time`。

#### 8.3.3 POST `/api/v1/evaluation` 与 GET `/api/v1/evaluation/{taskId}`

同预测模式。`EvaluationResultVO` 在 `completed` 时返回 `metrics`（`Map<String,Double>`），`failed` 时返回 `errorMessage`。

#### 8.3.4 僵尸任务恢复

服务重启后可能残留 `status=processing` 的僵尸记录，由定时任务每 60 秒扫描 `update_time < NOW() - INTERVAL 10 MINUTE` 的记录标记为 `failed`，详见 [任务管理/后端实现.md](../03-模块设计/基础模块/任务管理/后端实现.md)。

---

## 9. 错误处理规范

### 9.1 前端处理建议

```typescript
// 响应拦截器示例
axios.interceptors.response.use(
  (response) => {
    const { code, msg, data } = response.data;
    if (code === '00000') {
      return data;
    }
    // 业务错误处理
    handleBusinessError(code, msg);
    return Promise.reject(new Error(msg));
  },
  (error) => {
    // HTTP 错误处理
    const status = error.response?.status;
    if (status === 401) {
      // Session 失效，跳转登录
      redirectToLogin();
    }
    return Promise.reject(error);
  }
);

function handleBusinessError(code: string, msg: string) {
  if (code.startsWith('A02')) {
    // 认证错误
    message.error(msg);
    redirectToLogin();
  } else if (code.startsWith('A03')) {
    // 权限错误
    message.warning('您没有权限执行此操作');
  } else if (code.startsWith('A04')) {
    // 参数错误
    message.error(msg);
  } else {
    // 其他错误
    message.error(msg || '系统繁忙，请稍后重试');
  }
}
```

### 9.2 后端异常处理

系统统一使用全局异常处理器，确保所有异常返回标准响应格式。

---

## 10. 版本管理

### 10.1 版本策略

- 当前版本：`v1`
- 版本号包含在 URL 路径中：`/api/v1/xxx`
- 重大变更发布新版本（如 `v2`），旧版本保留兼容期

### 10.2 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0 | 2024-01-01 | 初始版本，包含基础模块 API |

---

## 11. 附录

### 11.1 接口详情查询

完整的接口参数、Schema 定义可通过 OpenAPI MCP 工具获取：

```bash
# 获取 OpenAPI 规范概览
mcp_call_tool(serverName="API文档", toolName="read_project_oas_yfcdew")

# 获取具体接口详情
mcp_call_tool(serverName="API文档", toolName="read_project_oas_ref_resources_yfcdew", 
              arguments={"path": ["/paths/_api_v1_users_page.json"]})
```

### 11.2 状态码代码定义

状态码枚举定义位于：`dehaze-java/src/main/java/com/pei/dehaze/common/result/ResultCode.java`

### 11.3 响应结构代码定义

- 基础响应：`dehaze-java/src/main/java/com/pei/dehaze/common/result/Result.java`
- 分页响应：`dehaze-java/src/main/java/com/pei/dehaze/common/result/PageResult.java`
