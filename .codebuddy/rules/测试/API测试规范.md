---
# 注意不要修改本文头文件，如修改，CodeBuddy（内网版）将按照默认逻辑设置
type: always
---
# dehaze-sdk-js API 测试编写规范

## 测试代码结构规范

- 测试文件位置：`test/modules/<domain>/<api>.test.ts`
- 推荐骨架：
    - 顶层 `describe("<模块名> 接口测试", () => { ... })`
    - 在模块级 `describe` 中使用 `beforeAll(login)` / `afterAll(logout)` 统一管理登录态
    - 每个 API 使用一个子 `describe("METHOD PATH - 描述", () => { ... })`
    - 每个 `test()` 严格遵循
        - Arrange：准备数据（使用 factories）、准备权限/前置状态
        - Act：调用 SDK（如 `UserAPI.xxx()`）
        - Assert：强断言关键字段与业务逻辑

参考：`test/modules/user/user.test.ts` 当前风格可作为范例，但应避免过度"循环断言"导致噪音（只断言关键字段即可）

## 鉴权与会话管理

- 需要登录态的模块测试：在 `describe` 的 `beforeAll` 里 `login()`，在 `afterAll` 里 `logout()`
- 禁止在每个 test 里重复登录/登出（会增加不稳定性与耗时）。

## 测试数据"可重复性"约束

### 数据隔离

- **谁创建谁清理**：测试套件创建的数据必须在 `afterAll` 统一清理
- 清理应以"记录创建的 ID 列表"为准（如 `createdUserIds: number[]`），而不是依赖模糊查询

### 唯一标识生成

- 必须使用 `test/factories/common.ts` 提供的 `uniqueName/uniqueEmail/uniqueMobile` 等生成器
- 禁止仅依赖 `faker.seed` 生成唯一数据（跨运行可能重复）；唯一性必须包含时间戳/计数器等机制（当前 `uniqueMobile/uniqueEmail`
  已满足）

### 固定测试常量

- 对"预置数据"的断言（如 admin/test 用户、角色、部门）必须来自 `test/factories/constants.ts`，禁止在用例里散落硬编码

### 通过 factories 生成测试数据：

- 表单/查询：例如 `createUserForm/createUserQuery` 等
- 唯一字段：必须使用 `uniqueName/uniqueEmail/uniqueMobile` 等生成器

## 断言规范

- 必须断言关键字段：如 `id`、`username`、`status`、权限/角色等
- 优先使用项目内断言工具，如 `expectBizErrorOrUndefined(promise, ["A0400", "B0001"])`
- 对列表分页：
    - 必须断言结构：`list` 是数组、`total` 是 number、`list.length <= pageSize`
    - 必须断言至少一条"确定存在"的样本（例如预置 admin 用户可见性）
- 对文件/二进制：
    - 必须断言类型（`ArrayBuffer`/`Buffer`）与合理大小（避免只断言 not null）
    - 如涉及强业务逻辑，excel导出、导入，需校验文件格式、内容、关键字段

> 注意：错误码断言推荐使用项目内断言工具（如 `expectBizErrorOrUndefined`），并允许后端存在差异时列出"可接受错误码集合"