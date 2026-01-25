## 初步错误分类

根据响应状态码与现象一般问题可能出现再以下范围内，此阶段仅做定位，不急于修改 SDK 或测试代码：

- 测试代码编写错误（编译报错、指向前端的运行错误、用例结构、并发、生命周期、异步等待、mock/环境等）
- 测试数据与断言错误（数据构造、清理、唯一性、断言过强/过弱/断言字段不稳定）
- SDK 封装问题（params vs data、path 拼接、header/cookie、序列化、响应解包）
- 后端代码问题（接口实现、权限、校验、业务逻辑、并发/幂等、异常处理）
- 数据库/缓存问题（脏数据、唯一索引冲突、事务/隔离级别、缓存未失效、环境库错用）

## 测试失败分析

每次失败先收集一组信息：

1) **失败形态**

- 稳定失败（每次必现）
- 偶现/抖动（同代码同环境有时过有时不过）
- 行为异常（200 但数据不对；或返回结构变化；或字段偶尔缺失）

2) **最小事实**

- 失败接口（method + path）
- 请求体/查询参数（最终发出去的那份）
- 响应（HTTP status、业务 code/message、data 结构）
- 用例并发信息（是否多个用例同时创建/修改同一资源）

3) **对照基准**

- OAS/接口文档期望（参数位置、必填、类型、返回结构）
- 用同 token 的 curl 结果（成功/失败 & 结果是否合理）

## 详细排查步骤

### Step 1：curl 复现

在同环境、同 token、同参数下请求后端，分析问题来源。绕过 SDK 与测试代码，直接调用后端接口：

- curl 也失败 → 权限/参数传递错误、后端错误、DB/缓存错误、判断失败是否符合业务逻辑
- curl 成功且结果合理，但测试失败 → 优先怀疑 SDK 封装或测试逻辑问题

调用示例：

```bash
# 获取 token
TOKEN=$(curl -s -X POST "http://localhost:${API_PORT:-8989}/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=123456" | jq -r '.data.accessToken')

# 测试 API，查看完整响应
curl -s -X POST "http://localhost:${API_PORT:-8989}/api/v1/users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","nickname":"测试","roleIds":[3],"deptId":1}' | jq '.'
```

环境与多后端切换约定：

- 8989：Java/Spring Boot 后端（`dehaze-java`）。
- 8999：Go/Gin 后端（`dehaze-go`）。
- 5000：Python/FastAPI 后端（`dehaze-python`）。
- 通过环境变量 `API_PORT` 切换目标后端；排查时应明确当前指向哪个实现。
- 当怀疑“环境差异”导致失败时：第一步仍是 curl 对比两个环境的真实响应，再决定是 SDK/测试还是后端差异。

### Step 2：获取 OAS/接口文档

对失败接口从 OAS/接口文档中核对以下要素。

- HTTP method：GET/POST/PUT/PATCH 是否一致。
- 参数位置：字段在 query/body/path 中的定义是否与当前调用一致。
- required：必填字段是否完整传递。
- pattern/format：邮箱、手机号等格式约束是否满足。

发现测试或 SDK 与文档不一致 → 应优先调整测试/SDK 以对齐文档

- API 文档要求 query，但你发 body（或相反） →
- 可能错误地方：当前测试用例问题
- SDK 封装和文档不一致
- 文档返回数据结构是 `{ code, message, data }`，你断言了 `res.data.data.xxx` 这种多解包 →
- 可能错误地方：断言错误、响应解包错误
- 文档写明必填，你没传 →
- 可能错误地方：测试用例写错/数据工厂漏字段、SDK字段被过滤/重命名
  文档与后端实现不符 → 在归因中标注为后端/文档不一致问题。

API文档调用示例：

```typescript
// 1. 先读取 OAS 文件列表
mcp_call_tool("API文档", "read_project_oas_at44y9", {})

// 2. 读取具体接口定义，检查参数位置
mcp_call_tool("API文档", "read_project_oas_ref_resources_at44y9", {
  path: ["/paths/_api_v1_users.json"]
})
```

### Step 3：检查 SDK 封装

重点排查 dehaze-sdk-js 中的封装是否存在以下问题：

- GET 请求是否错误地使用了 `data` 而非 `params`
- POST/PUT/PATCH 请求是否错误地使用了 `params` 而非请求体
- path 参数拼接是否正确（如 `/api/v1/users/{id}` 是否被正确替换为具体 ID）
- header/cookie 没带上（鉴权 token 未注入、CSRF/cookie 丢）
- content-type 错（本该 JSON 却 form；或上传/二进制处理错误）
- 返回值解析是否正确（是否错误地从错误层级读取 `data`/`code`/`message`）

若 curl 与 OAS 显示后端行为正确但测试失败，应优先在此步骤定位并修复封装问题。重点判断同一个 token、同一业务意图，**相同请求是否一致
**。

### Step 4：分析测试用例自身问题

分析问题来源可能是并发/生命周期/等待/清理，把“测试写法问题”与“数据/断言问题”拆开。

**A 测试代码编写错误（结构/时序/隔离）常见信号：**

- 忘了 `await`（请求没完成就断言/退出）
- 测试依赖执行顺序（单跑通过、全量跑失败）
- `beforeAll`/`afterAll` 顺序不对，或共享了可变全局变量
- 清理逻辑跑在断言前，或清理失败不报错
- 检查 `afterAll` 清理逻辑是否覆盖所有创建的 ID，避免前一次运行残留数据影响本次测试
- 多个用例并发改同一个资源（同用户名/手机号/同一条记录），导致互相污染

**B 测试数据与断言错误常见信号：**

- 使用非唯一数据（导致重复键冲突、或取到别人用例的数据）确认是否使用 `uniqueName/uniqueEmail/uniqueMobile` 等生成唯一数据
- 断言了不稳定字段：`createTime/updateTime/id`、排序未固定的列表、随机生成内容
- 断言过弱（只 `toBeDefined` 导致漏检）或过强（后端允许字段为空但你强制非空）
- 用错了“预期业务错误”的断言助手（把应失败当成功/反之）

快速判别：
> - **只在全量/并发时失败** → 优先 A/B（隔离、唯一性、清理）
> - **单接口稳定失败** → 优先 SDK 封装问题 / 后端代码逻辑有误（请求构造或后端逻辑）

### Step 5：后端/DB

当 Step1~4 都排除后，再深入分析，区分后端代码错误还是数据库/缓存问题。

- 使用 SQL 查询是否存在重复数据、脏数据或唯一约束冲突。
- 使用 Redis 命令检查缓存是否过期/污染，必要时清理对应 key。
- 如无 DB/Redis 工具，可要求用户提供查询结果或后端日志片段，再据此给出判断。

```typescript
// 检查用户缓存
mcp_call_tool("redis", "scan_keys", { pattern: "user:*" })

// 查看具体缓存内容
mcp_call_tool("redis", "hgetall", { name: "user:info:123" })

// 清理测试缓存
mcp_call_tool("redis", "delete", { key: "user:info:123" })
```

**D 后端代码问题常见信号：**

- curl 稳定失败，且错误与参数无关（500、空指针、业务 code 异常）
- 相同请求在不同时间返回不一致（业务状态机/并发处理缺陷）
- 权限/鉴权逻辑与文档不一致（应 403 却 200 或反之）
- 返回结构与 OAS 不一致（字段缺失/类型变化）

**E 数据库/缓存问题常见信号：**

- duplicate key / 唯一索引冲突（尤其偶现，常由历史脏数据或并发导致）
- 删不干净/查到旧数据（缓存未失效、读写分离延迟、事务未提交）
- 环境库用错（连到共享库/非测试库，数据被别人污染）
- 同请求第一次失败、重试成功（强烈指向缓存/一致性/事务时序）

示例：

```typescript
// 查看测试残留数据
mcp_call_tool("mysql", "execute_sql", {
    sql: "SELECT id, username, mobile, email FROM sys_user WHERE username LIKE 'test%'"
})

// 检查唯一约束冲突
mcp_call_tool("mysql", "execute_sql", {
    sql: "SELECT mobile, COUNT(*) as cnt FROM sys_user GROUP BY mobile HAVING cnt > 1"
})

// 清理测试数据（谨慎使用）
mcp_call_tool("mysql", "execute_sql", {
    sql: "DELETE FROM sys_user WHERE username LIKE 'testuser_%'"
})
```

### Step 6：归因结论输出

根据前述步骤形成归因结论与下一步建议，遵循下表：

| 场景                | 归因            | 下一步              |
|-------------------|---------------|------------------|
| curl 成功，测试失败      | SDK 封装或测试代码问题 | 修复 SDK/测试数据与断言   |
| curl 失败且符合业务逻辑    | 测试预期错误        | 调整测试预期与错误码集合     |
| curl 失败且违反业务逻辑    | 后端 bug        | 编写并保留失败测试 + 注释模板 |
| curl 与测试都成功但行为不合理 | 后端 bug        | 同上，补充业务期望说明      |
