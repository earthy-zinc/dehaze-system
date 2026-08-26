---
description: dehaze-python 单元测试编写强制规范（2026-08-23 测试架构重构后）。在 dehaze-python/tests/ 下新增或修改任何测试、编写 conftest fixture、为服务补构造注入时必须遵守
alwaysApply: false
enabled: true
updatedAt: 2026-08-23T00:00:00.000Z
provider:
---

# dehaze-python 测试规范

1. **数据库**：用 conftest 的 `db` fixture（真实 MySQL 测试库 dehaze_test，SAVEPOINT 事务回滚）；禁止 StubAsyncSession/NullDBSession 做传递占位（故障注入除外）。
2. **Redis**：`mock_redis` 为 autouse 全局接管，禁止手动 patch get_redis_client；需断言状态时显式请求该 fixture 拿同一实例。
3. **Mongo**：用 `mongo_db` fixture（mongomock-motor），仓储已单点 patch。
4. **仓储依赖**：服务类构造注入（`__init__` 参数 + 模块单例默认值），测试注入 `SimpleNamespace(get_x=AsyncMock(...))` 显式配方法（注意 AsyncMock(return_value=X) 的子方法返回子 mock 而非 X）；禁止 monkeypatch 模块属性替换仓储。
5. **断言**：只断言业务结果（落库状态/返回值/真实文件效果），禁止断言 mock 调用序列。
6. **外部 HTTP**（LLM/连通性）：respx 且必须 `assert_all_mocked=True`；ES 在 index 模块函数边界 mock（aiohttp 拦不到）；存储注入 `LocalStorageService(tmp_path)`。
7. **桩**：从 `tests/stubs/{fakes,factories,mocks}` 按职责导入。
8. **目录**：镜像源码包路径（app/service/ai/service/ → tests/service/ai/service/）。
9. **Marker**：router 测试→`api`；db fixture 用户→`requires_db`；call 实测 >2s→`slow`。
10. **运行**：`uv run pytest`（自动 APP_ENV=testing）；多进程/多 agent 并行须 `DB_NAME=xxx` env 隔离。
11. **命名**：`test_功能_场景`，禁止 test_xxx_gaps/test_xxx_repro 补丁式堆积。
12. 规范全文与示例见 `dehaze-doc/docs/02-系统架构/05-测试架构.md` §6.4。
