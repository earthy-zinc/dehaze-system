# SDK 集成测试

动手前确认：被测后端在跑、数据库连通、`test/config` 与 `test/factories` 就绪。集成测试用 `BACKEND_URL` 指定被测后端，写命令时用 `cross-env BACKEND_URL=<url>` 作前缀。

## 流程

1. 问清需求。信息不足就主动问四件事：这个接口在业务流程里起什么作用、核心功能是什么；测哪个接口，一次聚焦一个主接口；调它需要什么前置条件，比如登录态、权限、特定数据；这个场景下哪些边界重要。一个接口可以设计多个相关用例，比如成功路径与参数校验，但不要同时为多个不相关接口写；用例超过 10 个就分批写、分批验。

2. 列用例清单，每个接口至少覆盖这些类别：

   | 类别 | 覆盖要点 | 输出要求 |
   | --- | --- | --- |
   | 成功路径 | 响应结构、关键字段、核心业务语义 | 验证返回完整性与业务正确性 |
   | 参数校验 | 必填缺失、格式非法、边界值 | 用 `expectBizErrorOrUndefined` 断言错误码 |
   | 权限与鉴权 | 未登录、无权限访问 | 验证 401 / 403 等鉴权响应 |
   | 幂等与一致性 | 相同条件重复调用结果一致 | 针对查询类、无副作用接口 |
   | 列表与分页 | 分页参数、排序、过滤 | 验证 list 结构、total、pageSize |
   | 并发与隔离 | 多用例同时执行无冲突 | 确保数据唯一性与清理策略 |
   | 安全性 | SQL 注入、XSS、越权等 | 验证安全防护机制，如适用 |

3. 写用例。先查 SDK 与目标 API 的接口文档、`test/factories/` 里可用的工厂函数、`test/factories/constants.ts` 的预置数据。一次写一个完整 `test()` 块，相关用例可放同一 `describe()`；写完一个就跑 lint 与验证，别跳过去写下一个。

4. 过 lint。每写完一个用例立即跑 `cd clarity-sdk-js && pnpm run lint`。lint 之前别手动改能被自动修复的格式问题；确保没有语法或类型错误再继续；lint 报的错若影响测试逻辑，先修再走。

5. 跑用例。只跑当前写的用例。

   ```bash
   # 单个用例
   npx vitest run test/modules/<domain>/<api>.test.ts --testNamePattern="用例名称"

   # 整个文件
   npx vitest run test/modules/<domain>/<api>.test.ts

   # 全量，按后端分
   pnpm test:java     # BACKEND_URL=http://127.0.0.1:8989
   pnpm test:go       # http://127.0.0.1:8990
   pnpm test:python   # http://127.0.0.1:8991
   ```

   超时先查测试数据与后端响应时间，别盲目加 timeout。

6. 失败排查。先快速诊断：稳定失败还是偶现、错误在 HTTP 状态码还是业务码还是断言、有没有并发相关。再按 SDK 封装、测试代码、后端与 DB 的顺序定位，各类信号见 `references/sdk-排查手册.md`。归因后分流：测试问题改测试数据或断言；SDK 问题改 SDK 或调整调用方式；后端 bug 保留能暴露问题的失败用例，用注释说明预期行为与实际行为，禁用 `test.skip()` 或注释掉断言。

7. 回到清单取下一个用例，重复 3 到 6，直到全部写完通过，最后跑整个文件确认无回归。

## 基建与硬约束

- 套件数据隔离：测试文件尽量能并行跑，`afterAll` 只删本套件创建的资源，用 `datasetCreated` 这类标记做守卫。
- SDK 集成测试基建事实：vitest `fileParallelism:false`（并行互踢 A0230）；`expectBizError` 严格模式（业务码必须精确命中）；登录失败 5 次锁 30 分钟；改权限后清 `role:perms:*` 并重新登录；`test/utils/quota.ts` 自愈配额；离线单测放 `test/unit/`；`tsconfig` include `test/**` + strict + `exactOptionalPropertyTypes` ⇒ `npx tsc --noEmit` 校验契约；Stop hook 以指纹去重、只报新增；但 `tsc` 是项目级、无法按文件裁剪，"某错是否本次改动引入"仍须按文件/基线过滤判定。
- SDK 集成测试并行的硬约束（并行方案据此设计）：`member` / `member-edge` / `order` / `order-edge` / `package` / `billing` / `billing-edge` / `ai-schedule` / `knowledge-base` 会改写全局权益/等级/配额，必须同组串行；18 处"性能烟测"（200/500/1000/2000ms 阈值）依赖空载，并行或叠加负载必然 flaky ⇒ 须剥离到独立串行套件；登录侧不撞防重复（`/api/v1/auth` 前缀豁免）、不互踢（`USE_MULTI_POINT` 默认 False）。
- SDK 集成测试的基建事实：`test/config/vitest.setup.ts` 的全局 `beforeAll` 调 `login()`，其用户名默认 admin 且无环境变量开关（`test/utils/auth.ts`）⇒ 跑任一 SDK 测试文件都会以 admin 登录入队，会挤掉同账号的并行持有者会话（会话队列排查见 `project-knowledge` 的 `references/04-协作与排查纪律.md` 排查方法论）。
- Node undici 会 percent-encode 上传文件名：SDK 层测不出"文件名含控制字符"（`\x00-\x1f` 等）的真实行为，这类对抗语料必须由 pytest 层覆盖。
- 复跑必须带时间戳 + 类集清单，且排在所有域最后编辑之后；`brief.json`/日志 mtime＝结束时刻，归属须用 `/tmp/runN_started_at` 标记；`brief.json` 失败分布文件名带行号（如 `menu.test.ts:43`）⇒ 必须先 `split(':')` 再匹配。
- 输出格式、NDJSON 日志查询与运行期进度见 `references/sdk-运行与日志.md`。
