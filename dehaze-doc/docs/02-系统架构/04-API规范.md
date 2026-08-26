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

### 2.5 模块基础路径命名约定

#### 2.5.1 命名规则

| 规则 | 说明 | 示例 |
|------|------|------|
| 统一前缀 | 所有模块路径以 `/api/v1` 开头 | `/api/v1/users` |
| 模块名 kebab-case | 多词模块用连字符连接，全小写 | `image-input`、`ai-billing` |
| 资源名用复数 | 资源集合使用英文复数形式 | `users`、`roles`、`files` |
| 资源嵌套深度 ≤ 2 层 | 基础路径最多包含 2 个资源段 | `/api/v1/ai/conversations`（允许）<br>`/api/v1/ai/conversations/{id}/messages`（违规，应改为扁平路径如 `/api/v1/ai/messages?conversationId={id}`） |
| 历史缩写保留 | `dept`、`sys`、`kb` 等历史通用缩写保留，新增模块不使用缩写 | `/api/v1/depts`（历史保留） |

#### 2.5.2 模块基础路径对照表

| 模块 | 基础路径 | 说明 |
|------|---------|------|
| 认证管理 | `/api/v1/auth` | |
| 用户管理 | `/api/v1/users` | |
| 角色管理 | `/api/v1/roles` | |
| 部门管理 | `/api/v1/depts` | 历史保留缩写 |
| 菜单管理 | `/api/v1/menus` | |
| 文件管理 | `/api/v1/files` | |
| 消息通知 | `/api/v1/notifications` | |
| 收藏管理 | `/api/v1/favorites` | |
| 反馈评价 | `/api/v1/feedbacks` | |
| 推荐管理 | `/api/v1/recommendations` | |
| 字典管理 | `/api/v1/dicts` | |
| AI知识库 | `/api/v1/kb` | |
| 语音交互 | `/api/v1/voice` | |
| AI计费管理 | `/api/v1/ai-billing` | |
| 订单管理 | `/api/v1/orders` | |
| 套餐管理 | `/api/v1/packages` | |
| 会员管理 | `/api/v1/members` | |
| 任务管理 | `/api/v1/tasks` | |
| 图像输入 | `/api/v1/image-input` | |
| 去雾处理 | `/api/v1/prediction` | |
| 效果对比 | `/api/v1/evaluation`、`/api/v1/compare` | |
| 算法选择 | `/api/v1/algorithms/select` | |
| 算法管理 | `/api/v1/algorithms` | |
| 数据集管理 | `/api/v1/datasets` | |
| AI对话 | `/api/v1/ai`（内部）、`/api/v1/chat`（OpenAI 兼容）、`/api/v1/messages`（Claude 兼容） | 双轨设计 |
| MCP能力网关 | `/api/v1/mcp` | JSON-RPC over HTTP |

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

### 5.3 模块错误码段位分配

为避免各模块错误码冲突，按模块分配错误码段位。错误码以 `A0`/`B0`/`C0` 开头：A 类为用户端错误、B 类为系统端错误、C 类为第三方错误。

#### 5.3.1 通用错误码（全局）

| 错误码 | 语义 |
|--------|------|
| `A0230` | token 无效 |
| `A0231`~`A0239` | 超级管理员保护系列、会话相关 |
| `A0400` | 参数错误 |
| `A0401` | 资源不存在 |
| `A0403` | 无权限 |
| `A0500` | 业务异常 |
| `A0501` | 数据已存在 |
| `A0502` | 数据状态不允许 |
| `A0503` | 操作不允许 |
| `A0504` | 业务错误 |

#### 5.3.2 模块段位分配表

| 段位范围 | 模块 | 说明 |
|---------|------|------|
| `A0110`~`A0119` | 用户管理 | 用户名已存在等 |
| `A0200`~`A0214` | 认证管理 | 登录认证、验证码相关 |
| `A0250`~`A0259` | 推荐管理 | |
| `A0300`~`A0309` | 文件管理 | 文件上传相关（`A0301` 无权限） |
| `A0400`~`A0409` | 通用资源不存在 | `A0401` 资源不存在 |
| `A0500`~`A0509` | 通用业务错误 | |
| `A0510`~`A0519` | 会员管理 | |
| `A0520`~`A0529` | 任务管理 | |
| `A0530`~`A053F` | 订单管理 | 订单业务错误 |
| `A0540`~`A0549` | AI对话 | |
| `A0550`~`A055F` | 消息通知 | |
| `A0560`~`A056F` | 反馈评价 | |
| `A0570`~`A0579` | 套餐管理 | |
| `A0600`~`A0609` | AI 调用失败 | LLM 调用、流式超时、工具调用失败 |
| `A0610`~`A0619` | 算法管理 | |
| `A0620`~`A062F` | 算法选择 | |
| `A0630`~`A063F` | 去雾处理 | |
| `A0640`~`A064F` | 效果对比 | |
| `A0650`~`A065F` | 数据集管理 | |
| `A0660`~`A066F` | AI知识库 | |
| `A0670`~`A067F` | 语音交互 | |
| `A0680`~`A068F` | AI计费管理 | `A0680` 退款申请已存在、`A0681` 退款审核失败、`A0682` 配额不足/欠费熔断 |
| `A0690`~`A069F` | MCP能力网关 | |
| `A0700`~`A070F` | 图像输入 | 上传格式/大小/分辨率错误 |
| `B0001`~`B000F` | 通用系统错误 | |
| `B0100`~`B010F` | 系统执行超时 | 统一含义：流式输出超时、长任务超时 |
| `B0300`~`B030F` | 任务管理 | 任务执行错误 |
| `C0001`~`C000F` | 第三方服务 | 支付、存储、LLM 提供商等 |

#### 5.3.3 段位使用约束

- `B0100` 统一含义为"系统执行超时"，不得用于"数据集不存在"等其他语义。图像输入模块历史曾将 `B0100` 用于"数据集不存在"的语义冲突场景，新增接口表达资源不存在必须使用 `A0401`。
- `A0230`~`A0239` 保留为"超级管理员保护系列"和"会话相关"，其他模块不得占用。
- 模块新增错误码必须落在分配段位内，超出段位需在本文档登记后再使用。
- 通用错误码（`A0400`、`A0401`、`A0403`、`A0500`~`A0504`）所有模块通用，无需重复定义。

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

服务端统一使用 **Asia/Shanghai (UTC+8)** 时区，三端 + 数据库 + 连接层显式固化，不依赖部署系统时区（方案B）：
- **Python**：数据库连接 `init_command=SET time_zone='+08:00'`（`app/config.py` DATABASE_URL）；审计时间写入显式 Asia/Shanghai（`app/models/base.py` `_shanghai_now()`）
- **Go**：MySQL DSN `loc=Asia/Shanghai`（`pkg/database/config.go`），DATETIME 读回按该时区解释
- **Java**：JDBC `serverTimezone=Asia/Shanghai` + Jackson `time-zone: GMT+8`（`application-*.yml`）
- **数据库服务器**：部署要求 MySQL `default-time-zone='+08:00'`（当前实例 `SYSTEM`=CST，等价）

时间字段以 **naive datetime**（无时区标记，`yyyy-MM-dd HH:mm:ss`）存储与返回，语义即 Asia/Shanghai。

**前端解析约定**：收到的时间字符串一律按 Asia/Shanghai 解析（`yyyy-MM-dd HH:mm:ss`），如需用户本地时区展示由前端自行换算，**不得按浏览器本地时区直接解析 naive 字符串**（否则非 UTC+8 用户显示错 8 小时）。

**部署要求**：服务容器/进程 `TZ=Asia/Shanghai`（Java 进程建议 `-Duser.timezone=Asia/Shanghai`），保证 `datetime.now()` / `time.Now()` / `LocalDateTime.now()` 本地时间语义一致。日志时间戳（ISO8601 UTC）为运维时间线，与业务时间字段分离，不参与业务计算。

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

### 7.2 权限标识命名约定

#### 7.2.1 命名格式

权限标识采用 3 段式：`module:resource:action`，全小写，以冒号分隔。

| 段位 | 说明 | 示例 |
|------|------|------|
| `module` | 模块缩写 | `sys`、`user`、`role`、`dept`、`menu`、`file`、`notify`、`kb`、`voice`、`ai`、`mcp`、`recommend`、`feedback`、`order`、`member`、`task`、`dehaze`、`evaluation`、`image`、`dataset`、`algorithm` |
| `resource` | 资源名（小写英文） | `user`、`role`、`model`、`tool`、`document` |
| `action` | 动作 | `list`、`view`、`add`、`edit`、`delete`、`export`、`import`、`manage`、`execute`、`audit`、`adjust`、`stat`、`refund`、`kick`、`reset`、`password` |

#### 7.2.2 特殊接口标识

| 接口类型 | 权限标识 | 说明 |
|---------|---------|------|
| 公开接口（无需登录） | `-` | 在接口文档中显式标注 |
| 登录态接口（无需特殊权限） | `-` | 在权限标识汇总中统一说明"仅需登录" |
| 管理员专用 | `module:resource:manage` 或具体动作 | |

#### 7.2.3 正反例

| 类型 | 标识 | 说明 |
|------|------|------|
| ✅ 正例 | `sys:user:add` | 标准 3 段式 |
| ✅ 正例 | `ai:model:manage` | 管理员专用 |
| ✅ 正例 | `mcp:tool:manage` | MCP 工具管理 |
| ✅ 正例 | `recommend:rule:view` | 推荐规则查看 |
| ❌ 反例 | `sys:recommendation:rule:view` | 4 段式，违规 |
| ❌ 反例 | `ROLE_CODE_EXISTS` | 错误码当权限标识 |
| ❌ 反例 | `管理员` | 文字描述，应使用权限码 |

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
