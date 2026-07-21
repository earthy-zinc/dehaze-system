---
name: dehaze-coder
description: >-
  dehaze-system 项目的统一编码与测试规范 skill。当用户需要为 dehaze-system 写组件、写接口、写 API、实现需求、生成代码、修改代码、添加功能、修复 bug、重构代码、编写测试，或涉及 dehaze-system 项目的任何代码编写、代码修改、功能实现、技术实现任务时触发此 skill。
  适用于 dehaze-system monorepo 中所有子项目：dehaze-go、dehaze-python、dehaze-java*、dehaze-front-vue、dehaze-front-react、dehaze-tool/dehaze-sdk-js 等。
  当用户提到"编码规范"、"测试规范"、"开发规范"、"编程规范"、"写代码"、"写接口"、"修 bug"、"重构"、"写测试"、"加功能"等关键词，且上下文涉及 dehaze-system 项目时，优先使用此 skill。
  同时涵盖 dehaze-sdk-js 的 Vitest API 集成测试编写、运行与失败排查。当用户提到"dehaze-tool/dehaze-sdk-js"、"SDK-JS"、"vitest"、"API 接口测试"、"测试失败排查"等场景时也触发此 skill。
---

# dehaze-coder：统一编码与测试规范

本 skill 为 dehaze-system monorepo 提供项目特定的编码与测试约束。通用的编码最佳实践（代码整洁、命名风格、设计模式等）不再重复，聚焦于项目特定的约定和工作流程。

## 项目文档查阅

开发前应查阅 `dehaze-doc` 中的相关文档，仅检索与当前任务直接相关的部分，避免全量阅读：

- **系统架构**：`dehaze-doc/docs/02-系统架构/` — 总体架构、环境兼容性、数据库设计、API 规范、测试架构、部署架构
- **模块设计**：`dehaze-doc/docs/03-模块设计/[模块]/` — 需求规格、API 接口、后端/前端实现、测试用例
- **项目实现**：`dehaze-doc/docs/04-项目实现/` — 各技术栈（Vue/React/Taro/Flutter/Android/Java/Go/Python）的架构文档

当代码改动影响了模块接口、数据流或架构设计时，同步更新 `dehaze-doc` 中对应的设计文档。

---

## 项目特定编码规范路由

根据当前操作的子项目，加载对应的 reference 文件以获取项目特定约束：

### Go 后端（dehaze-go）

- 编码：查阅 [Go 编码规范](./references/go-coding.md)

### Python 后端（dehaze-python）

- 编码：查阅 [Python 编码规范](./references/python-coding.md)

### Java 后端（dehaze-java\*）

- 编码：查阅 [Java 编程规范](./references/java-coding.md)

### 前端（dehaze-front-vue / dehaze-front-react）

前端项目遵循通用框架最佳实践，无额外项目特定规范。项目架构和组件设计查阅 `dehaze-doc/docs/04-项目实现/前端/` 下对应文档。

### dehaze-sdk-js（dehaze-tool/dehaze-sdk-js）

#### 环境与目录硬约束

**Vitest 运行环境：**

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

**目录与职责边界：**

- 用例文件：`dehaze-tool/dehaze-sdk-js/test/modules/**/**.test.ts`。
- 工厂函数：`dehaze-tool/dehaze-sdk-js/test/factories/**`：
  - 固定常量：`test/factories/constants.ts`（如预置 `USERS/ROLES/DEPTS`、固定可见用户数等）。
  - 通用唯一数据生成：`test/factories/common.ts`（如 `uniqueEmail/uniqueMobile/uniqueName`）。
  - 业务表单/查询对象工厂：如 `test/factories/user.ts` 中的 `createUserForm/createUserQuery` 等。

> 约束：测试文件内避免重复手写大量随机/拼接数据；优先复用 factories，保证一致性、唯一性与可复现排查路径。

#### 测试编写规范

- API 测试编写规范：查阅 [SDK-JS 测试编写规范](./references/sdk-js-testing.md)

#### Workflow A：编写/补充测试

"为某个 `UserAPI.xxx`/`RoleAPI.xxx` 等接口新增或补充测试" → 使用 Workflow A。

如在编写测试期间发现接口行为与预期不符，可在完成 A 流程的基础上切换到 Workflow B 进行归因，并在确认是后端问题后固化 bug 用例。

**【重要】**：编写测试时必须查看 [测试用例编写步骤](./references/sdk-js-write-test.md)

#### Workflow B：测试失败排查与归因

目标：当 dehaze-sdk-js API 测试失败、不稳定或行为异常时，按固定顺序判断问题来源（测试代码编写错误/测试数据与断言错误/SDK 封装问题/后端代码/数据库问题），并给出下一步改动建议。

当测试失败时，**必须**按照归因决策树逐步判断，**严禁**跳过任何步骤直接修改测试预期。

**归因决策树：**

```
测试失败
  │
  ├─ 明显语法/逻辑错误？ → 修复测试代码
  ├─ 违反测试规范？ → 修复测试代码
  │
  ├─ Curl 验证
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

详细排查步骤见 [测试失败排查与归因](./references/sdk-js-test-failure.md)，在非失败排查阶段，请勿查看。
