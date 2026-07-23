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
| `Authorization` | 是（鉴权接口） | JWT Token | `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
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
Authorization: Bearer <token>
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
| `A0230` | token无效或已过期 | JWT Token 失效 |
| `A0231` | token已被禁止访问 | Token 已加入黑名单 |

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

**文件上传 (A07xx)**

| code | msg | 说明 |
|------|-----|------|
| `A0700` | 用户上传文件异常 | 文件上传通用错误 |
| `A0701` | 用户上传文件类型不匹配 | 文件类型不支持 |
| `A0702` | 用户上传文件太大 | 文件大小超限 |
| `A0703` | 用户上传图片太大 | 图片大小超限 |

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

### 6.1 JWT Token 认证

系统采用 JWT（JSON Web Token）进行身份认证，Token 通过 `Authorization` 请求头传递。

**认证流程：**

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Server as 服务端
    participant Redis as Redis

    Client->>Server: POST /api/v1/auth/login (username, password)
    Server->>Server: 校验凭证
    Server->>Redis: 存储 Token 信息
    Server-->>Client: { accessToken, tokenType, expires }
    
    Client->>Server: GET /api/v1/auth/me (Authorization: Bearer <token>)
    Server->>Redis: 校验 Token 状态
    Server-->>Client: { code: "00000", data: {...} }
```

### 6.2 登录接口

**请求：**

```http
POST /api/v1/auth/login HTTP/1.1
Content-Type: application/x-www-form-urlencoded

username=admin&password=123456
```

**响应：**

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "tokenType": "Bearer",
    "expires": 3600
  }
}
```

### 6.3 Token 使用

所有需要认证的接口须在请求头携带 Token：

```http
GET /api/v1/auth/me HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 6.4 退出登录

```http
DELETE /api/v1/auth/logout HTTP/1.1
Authorization: Bearer <token>
```

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
| **认证中心** | `/api/v1/auth` | 登录、登出、验证码 |
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
| `/{module}/_export` | GET | 导出 | `{module}:export` |
| `/{module}/_import` | POST | 导入 | `{module}:import` |
| `/{module}/template` | GET | 下载模板 | - |

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
      // Token 失效，跳转登录
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
