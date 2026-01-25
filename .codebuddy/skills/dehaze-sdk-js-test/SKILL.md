---
name: dehaze-sdk-js-test
description: 为 dehaze-sdk-js 编写/重构/运行 Vitest API 集成测试的一套标准化的工作流。适用于编写基于 Vitest 和 dehaze-sdk-js的API集成测试，当API接口测试运行失败时如何排查问题，以及用户提到“代码路径 dehaze-tool/dehaze-sdk-js、SDK-JS、vitest、生成 API 接口测试、API接口测试失败排查”等场景。不适用于普通前端单测、其他语言后端测试（Java/Go/Python）或与 dehaze-sdk-js 无关的库。
---

# dehaze-sdk-js 测试与问题排查流程

## 适用范围与触发条件

- 适用对象：`dehaze-tool/dehaze-sdk-js` 仓库中的 **Vitest API 集成测试** 及其失败排查。
- 触发信号：当用户提到以下任一关键词时优先启用本 skill：
    - 代码路径：`dehaze-tool/dehaze-sdk-js`
    - 行为描述：
        - 编写/补充 API 测试 —— Workflow A
        - 排查 dehaze-sdk-js 测试失败、确认是否后端 bug 等 —— Workflow B
- 不适用场景：普通前端单测、其他语言后端测试（Java/Go/Python）或与 dehaze-sdk-js 无关的 SDK。
- 当测试失败时，**必须**按照归因决策树逐步判断，**严禁**跳过任何步骤直接修改测试预期。

---

## 环境与目录硬约束

### Vitest 运行环境

基于仓库的 Vitest 配置，测试环境具有以下特征：

- `environment: "node"`：测试在 Node 环境执行，不依赖浏览器 DOM。
- `include: ["test/**/*.test.ts"]`：仅该目录与命名规则下的文件会被执行。
- `setupFiles: ["./vitest.setup.ts"]`：登录等全局初始化逻辑集中在 setup 中。
- 配置了较大的 `testTimeout/hookTimeout`，但应优先通过 **数据准备与接口速度** 控制耗时，而非在用例里再叠加超大 timeout。
- 并发：`maxConcurrency: 10` + `fileParallelism: true` → 测试必须在 **并发执行下仍然稳定**，测试数据天然隔离且唯一。

据此，编写或修改测试时必须满足：

- 避免对执行顺序、全局可变状态有隐式依赖。
- 为每个用例生成独立且唯一的数据，避免跨用例/跨文件互相污染。
- 所有外部依赖（后端服务、DB/Redis 等）在并发访问下也应可重复执行。

### 目录与职责边界

- 用例文件：`dehaze-tool/dehaze-sdk-js/test/modules/**/**.test.ts`。
- 工厂函数：`dehaze-tool/dehaze-sdk-js/test/factories/**`：
    - 固定常量：`test/factories/constants.ts`（如预置 `USERS/ROLES/DEPTS`、固定可见用户数等）。
    - 通用唯一数据生成：`test/factories/common.ts`（如 `uniqueEmail/uniqueMobile/uniqueName`）。
    - 业务表单/查询对象工厂：如 `test/factories/user.ts` 中的 `createUserForm/createUserQuery` 等。

> 约束：测试文件内避免重复手写大量随机/拼接数据；优先复用 factories，保证一致性、唯一性与可复现排查路径。

## Workflow A：编写/补充测试（标准流程）

- “为某个 `UserAPI.xxx`/`RoleAPI.xxx` 等接口新增或补充测试” → 使用 Workflow A。
- 如在编写测试期间发现接口行为与预期不符，可在完成 A 流程的基础上切换到 Workflow B 进行归因，并在确认是后端问题后固化 bug
  用例。

【重要】：编写测试时必须查看[测试用例编写步骤](./references/write-test.md)

## Workflow B：测试失败排查与归因（严格顺序）

目标：当 dehaze-sdk-js API 测试失败、不稳定或行为异常时，按固定顺序判断问题来源（测试代码编写错误/测试数据与断言错误/SDK
封装问题/后端代码/数据库问题），并给出下一步改动建议。

### 归因决策树
```
测试失败
  │
  ├─ 明显语法/逻辑错误？ → 修复测试代码
  ├─ 违反测试规范？ → 修复测试代码
  │
  ├─ CURL 验证
  │   ├─ curl 成功且符合预期 → 检查 SDK 封装
  │   ├─ curl 失败但符合业务逻辑 → 测试正确，无需修改
  │   ├─ curl 失败且违反业务逻辑 → 后端 bug ✅
  │   └─ curl 与测试不一致 → 检查 SDK 封装
  │
  ├─ SDK 封装问题？ → 修复 SDK
  │
  ├─ OAS 文档不一致？ → 对齐文档
  │
  ├─ 并发/环境/数据问题？ → 修复测试数据/清理
  │
  └─ 确认后端 bug → 保留失败测试 + 标记 + 反馈
```

详细的排查步骤见[测试失败排查与归因](./references/test-failure.md)，在非失败排查阶段，请勿查看
