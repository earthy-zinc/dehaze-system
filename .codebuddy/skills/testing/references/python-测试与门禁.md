# python 测试与门禁

clarity-python 的单测强约束（2026-08-23 测试架构重构后）与交付前的静态门禁，外加 python 侧的测试基建、已知噪音与举证纪律。

## 单测约束

1. 数据库：用 conftest 的 `db` fixture（真实 MySQL 测试库 clarity_test，SAVEPOINT 事务回滚）；禁止 StubAsyncSession/NullDBSession 做传递占位（故障注入除外）。
2. Redis：`mock_redis` 为 autouse 全局接管，禁止手动 patch get_redis_client；需断言状态时显式请求该 fixture 拿同一实例。
3. Mongo：用 `mongo_db` fixture（mongomock-motor），仓储已单点 patch。
4. 仓储依赖：服务类构造注入（`__init__` 参数 + 模块单例默认值），测试注入 `SimpleNamespace(get_x=AsyncMock(...))` 显式配方法（注意 AsyncMock(return_value=X) 的子方法返回子 mock 而非 X）；禁止 monkeypatch 模块属性替换仓储。
5. 断言：只断言业务结果（落库状态/返回值/真实文件效果），禁止断言 mock 调用序列。
6. 外部 HTTP（LLM/连通性）：respx 且必须 `assert_all_mocked=True`；ES 在 index 模块函数边界 mock（aiohttp 拦不到）；存储注入 `LocalStorageService(tmp_path)`。
7. 桩：从 `tests/stubs/{fakes,factories,mocks}` 按职责导入。
8. 目录：镜像源码包路径（app/service/ai/service/ → tests/service/ai/service/）。
9. Marker：router 测试→`api`；db fixture 用户→`requires_db`；call 实测 >2s→`slow`。
10. 运行：`uv run pytest`（自动 APP_ENV=testing）；多 worker/多 agent 并行用 `CLARITY_TEST_DB=xxx` 隔离库（默认 `clarity_test`），多 worker 加 `-n auto`。
11. 命名：`test_功能_场景`，禁止 test_xxx_gaps/test_xxx_repro 补丁式堆积。
12. 规范全文与示例见 `clarity-doc/docs/02-系统架构/05-测试架构.md` §6.4。

## check.sh 静态门禁

- 任何 Python 代码改动（`clarity-python`、`clarity-service/*`、`clarity-mcp-gateway`）交付前，必须通过终端跑通 `bash clarity-python/scripts/check.sh`，并在回报中给出脚本的实际退出码/输出作为证据
- 该脚本是唯一判定入口：IDE 的即时诊断不作为依据——本机 VS Code 的语言服务器可能未启用，且 pyright 的 typeshed 结论随版本变化，只有脚本内固定的 `PYRIGHT_VERSION` 才与 CI 口径一致
- 不要在 IDE 设置里另设一套类型检查口径（`python.analysis.typeCheckingMode`、`python.analysis.diagnosticSeverityOverrides`）：它既与 `pyrightconfig.json` 冲突被忽略，也会让 IDE 与门禁分裂
- 新增 Python 子项目时同步三处：子项目自身 `pyrightconfig.json`、仓库根 `pyrightconfig.json` 的 `include` + `executionEnvironments`、`check.sh` 里的 `PYRIGHT_PROJECTS`
- 禁止用文件级 `# type: ignore` / `# pyright: ignore`、拿 `Any`/`cast` 消音、或降低 severity 让检查变绿；抑制纪律与逐项目配置口径见 `clarity-doc/docs/02-系统架构/05-测试架构.md` §5.3

## 测试基建与已知噪音

- pytest（clarity-python）：`pytest.ini`＝`asyncio_mode=auto` / `testpaths=tests` / `--strict-markers`；conftest 每轮 DROP+CREATE `clarity_test` 并全量导入 `config/sql/{schema,data}/*.sql`（`sys_*.sql` glob 过滤）；`conftest.py` 必须在 import app 前强制 `os.environ` 覆盖测试库/端口（pydantic-settings 中 dotenv 优先于类默认值，否则会 DROP 开发库）；并发开发用 `CLARITY_TEST_DB=clarity_test_xxx` 隔离（含 conftest `GET_LOCK` 按库名区分；xdist `-n N` 下 conftest 自动派生 `{CLARITY_TEST_DB}_gw{i}` 每 worker 独立库，跨 worker 零行锁争用，worker 库可留存复用）；三个 `_run` 文件禁用 `asyncio.run()`。
- `clarity-mcp-gateway` 收集纪律：模块级禁副作用（勿在 import 期连真实网关或 `sys.exit`）；`pyproject.toml` 已配 `[tool.pytest.ini_options]`（防 `*_test.py` 误收），改动后用 `pytest --collect-only` 验证；live 校验手动跑 `tests/live_test.py`（连真实网关 8082，不入收集）。
- 按 md5 去重的上传类用例必须用每轮唯一内容（`f"…{uuid.uuid4()}".encode()`）：`upload_file` 命中残留 `sys_file` 行即秒传早返回，不再写本次 `tmp_path` 物理文件 ⇒ 物理断言假失败、或断言在残留命中时静默变弱。
- 语音流式 ASR 的孤儿会话会占并发名额：失败运行留下的会话堆在 `voice:asr:sessions`（30 分钟剪枝、上限 50），表现为后续运行全部"ASR 并发会话数已达上限"⇒ 先查 Redis（dev `db=1`）`zcard` 并清理。

## 质量红线

- 等待 fire-and-forget 后台 task 用 task 注册集合差集 + `await`（源码 `_pending_tasks` 模式），禁用 `asyncio.sleep`（"去 sleep"后须做变异验证）
- mock 的 SimpleNamespace 属性名必须与实体列名一致；"字段名错但默认值碰巧通过"＝契约错误须修；修 mock 目标必须是模块级 import（方法内局部 import 无法 mock）。

## 举证

- pytest 服务层断言不触发 `response_model` 校验 ⇒ VO 契约缺陷必须在 router/SDK 层覆盖（eval result 双重编码正是由 SDK 500 暴露）。
- 改断言前先跑 python 基准单文件确认真实行为（比读源码推演更快更准）；"测试断言与基准实现冲突"类（favorite 复活、dict 软删占键、package 删除后同名重建）按方案 a 改测试对齐 python 实际行为，不改实现。
- 修复引入行为变化（余额扣减/校验收紧）前先查测试数据前置依赖与自愈机制，否则"越修越挂"假回归；模块间依赖修复会"解锁"新失败面，勿把上轮失败清单当完备。
