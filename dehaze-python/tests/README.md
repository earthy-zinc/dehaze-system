# Dehaze Python 测试规范

本项目 Python 后端的测试规范与基础设施说明。本文档是**目标规范**：现有测试代码与规范存在差距的，按本文档逐步改造收敛（改造优先级见文末）。

## 1. 核心原则

1. **只替身"外部世界"，不 mock 内部逻辑**。数据库用真实 SQLite 内存库执行真实 SQL，Redis 用 fakeredis 执行真实协议；只有真正的第三方外部依赖（LLM、ES、外部 HTTP 服务）才做 mock。
2. **断言行为而非实现**。断言"调用返回了什么结果、状态产生了什么变化"，而不是断言"某个 mock 被调用了几次"。
3. **测试是资产不是补丁**。新增测试应补入对应模块的测试文件，禁止以 `test_xxx_gaps`、`test_xxx_repro`、`test_xxx_regression` 命名堆积补丁式文件；重构后整理原测试而非追加新文件。
4. **替身优先使用现成库**，禁止手写 `run_coro`/`async_ret`/自定义 Session 桩等反模式（见 §6、§8）。

## 2. 技术栈

| 依赖 | 用途 |
|------|------|
| pytest | 测试框架 |
| pytest-asyncio | async 测试（`asyncio_mode = auto`，无需装饰器） |
| pytest-mock | `mocker` fixture，统一 mock 管理（自动清理） |
| pytest-cov / pytest-xdist | 覆盖率 / 并行执行 |
| aiosqlite | SQLite 异步驱动，service 测试的真实数据库 |
| `fakeredis[lua]` | Redis 内存模拟（真实协议） |
| mongomock / mongomock-motor | MongoDB 内存模拟（motor 异步客户端兼容层） |
| respx | httpx 传输层拦截，mock 外部 HTTP 依赖（含 ES 8.x async 客户端） |
| freezegun | 时间冻结 |
| faker | 测试数据生成 |
| httpx | router 测试的 ASGITransport 客户端 |

## 3. 目录结构

```
tests/
├── conftest.py              # 共享 fixtures（db/mock_redis/mongo_db）与基础环境配置
├── stubs.py                 # 纯数据工厂（make_user_context 等），禁止 Session 桩/HTTP 假件
├── router/                  # HTTP 接口契约测试（ASGITransport 直连 FastAPI 应用）
├── service/                 # service 单元测试（主流，占绝大多数）
│   └── <module>/            # 子模块测试（如 ai/、billing/、import_export/）
├── repository/              # repository 层测试（真实 SQLite，验证 SQL 语义）
└── README.md                # 本文档
```

- 测试文件与被测模块一一对应：`app/service/billing/quota_service.py` → `tests/service/test_quota_service.py`
- 目录名与被测包路径一致，便于定位
- `stubs.py` 仅保留**纯数据工厂**（构造 `UserContext`/schema 对象，不涉及 I/O）；Session 桩、HTTP 假件、`run_coro` 等按 §8 反模式删除（P3 拆分前暂存于此）
- **环境变量必须在 conftest 顶部、import app 之前设置**（如 `os.environ["APP_ENV"] = "testing"`）：`app/config.py` 的 pydantic-settings 在首次 import 时读取环境变量并缓存为单例，测试文件若先 import app 再改环境变量将不生效

## 4. 测试层次

| 层次 | 测试对象 | 外部依赖处理 | 数量占比 |
|------|---------|-------------|---------|
| service 单测 | service 类方法 | SQLite 内存库 + fakeredis + respx | 绝大多数 |
| router 契约测试 | 路由 → 依赖覆盖 → 真实应用 | FastAPI `dependency_overrides` + ASGITransport | 少量（每路由组 1 个文件） |
| repository 测试 | SQL 语义/数据范围 | 真实 SQLite | 少量 |
| 集成测试（未来） | 真实中间件 | testcontainers | 极少量 |

### 4.1 router 测试模板（dependency_overrides 注入真实依赖）

router 测试通过 `app.dependency_overrides` 覆盖 FastAPI 依赖。**关键**：覆盖 `get_db` 时必须返回真实的 SQLite session（从 `db` fixture 取），不能返回 `object()`/`SimpleNamespace` 假 session——否则路由层把假对象塞进 `request.state.db`，业务代码拿到的不是内存库，测的是假链：

```python
# tests/router/test_xxx.py
@pytest.fixture
async def client(db, mocker):
    """ASGITransport 客户端：真实 db + 用户覆盖"""
    async def _override_db():
        return db  # 复用 5.1 的 db fixture session
    async def _override_user():
        return make_user_context(42)

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as c:
        yield c
    fastapi_app.dependency_overrides.clear()  # 必须清理，防泄漏到其他测试
```

- `get_db` 是 FastAPI 依赖（`app/database.py`），`get_current_user` 是鉴权依赖（`app/dependencies/auth.py`），都按需覆盖。
- 覆盖 `get_current_user` 用 `tests/stubs.py` 的 `make_user_context` 工厂，不要手写 dict/SimpleNamespace 冒充。
- 只测本路由组，直接 `from app.router import xxx` 导入，避免全量启动副作用。

## 5. 测试替身规范（核心）

### 5.1 数据库：SQLite 内存库，禁止手写 Session 桩

service 测试使用 `aiosqlite:///:memory:` 真实执行 SQL，**禁止** `StubAsyncSession`/`NullDBSession` 等手写 Session 假件（mock 掉数据库后测的是 mock 链，不是业务代码）。

```python
# tests/conftest.py
import app.database as database_module
from app.database import Base
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

@pytest.fixture
async def db(monkeypatch):
    """真实 SQLite 内存库（每测试独立，自动建表），并接管 get_db_session 入口"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,  # 必须：否则 create_all 与查询各连一个内存库 → no such table
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    # 必须：接管 service 内部 `async with get_db_session() as db` 的 session 来源，
    # 否则 fixture 与真实调用各用一套连接，fixture 形同虚设
    monkeypatch.setattr(database_module, "async_session_factory", factory)
    async with factory() as session:
        yield session
    await engine.dispose()
```

- `poolclass=StaticPool` 是硬性要求：SQLite 内存库每个连接独立，缺它会出现在建表连接上建表、在查询连接上查询的 `no such table`。
- `monkeypatch.setattr(database_module, "async_session_factory", factory)` 是 fixture 生效的关键：service 代码 98 处 `from app.database import get_db_session` + `async with get_db_session() as db`，必须让 `get_db_session` 内部引用的 `async_session_factory` 指向测试 factory，被测代码才会真正落到内存库。
- 建表使用 `from app.database import Base` 的 `Base.metadata.create_all`；不做迁移（Alembic 只用于真实环境）。

> 前置验证项：改造前须确认 `Base.metadata.create_all` 在 SQLite 上可完整建表（MySQL 特有列类型如 JSON 需映射兼容），如有不兼容列做一次性的类型映射调整。

### 5.2 Redis：fakeredis（已收敛到 conftest）

统一使用 `conftest.py` 的 `mock_redis` fixture，不重复 monkeypatch：

```python
async def test_login_success(db, mock_redis, ...):
    ...
    await mock_redis.setex("session:abc", 3600, json.dumps({...}))
    assert await mock_redis.get("session:abc")  # 断言真实状态
```

### 5.3 MongoDB：mongomock-motor

需要 Mongo 的模块（登录日志、审计日志、AI 调用日志）用 `AsyncMongoMockClient` 替代手写 dict 假件。**注意**：项目使用 motor 的 `AsyncIOMotorClient`，必须用 `mongomock-motor` 的兼容层（`mongomock.MongoClient` 是同步的，接口不兼容）：

```python
import mongomock_motor

@pytest.fixture
def mongo_db(monkeypatch):
    client = mongomock_motor.AsyncMongoMockClient()
    monkeypatch.setattr(mongo_module, "get_mongo_client", lambda: client)  # 替换模块级单例
    return client.db  # 支持 find/aggregate/count_documents/create_index 等
```

如 repository 已构造注入，则直接传入 `AsyncMongoMockClient().db`，无需 monkeypatch。

> **patch 点说明**：业务侧统一通过 `app/dependencies/mongo.py` 的 `get_mongo_db()` 依赖获取库（内部调 `get_mongo_client()`），patch `get_mongo_client` 即可覆盖所有使用方；不要 patch `_mongo_client` 私有单例。
>
> **mongomock 局限**：TTL 索引（`expireAfterSeconds`）不自动过期、聚合 `$lookup`/`$unwind` 等复杂管道支持有限、部分 update 操作符不完整。涉及这些特性的用例需在真实 Mongo（testcontainers）验证，或用 repository 层抽象隔离后单独测。

### 5.4 外部 HTTP（LLM / OCR / 下载）：respx

在 httpx 传输层拦截，替代手写 `FakeStreamResponse`/`FakeInternalResponse` 等 HTTP 假件：

```python
import respx
from httpx import Response

@respx.mock
async def test_llm_call():
    respx.post("https://api.llm.example/v1/chat").mock(
        return_value=Response(200, json={"choices": [...]})
    )
    result = await llm_service.chat(...)
    assert result == ...
```

**必须使用 `assert_all_mocked=True`**：respx 默认对未匹配请求放行（真实发出网络请求），漏 mock 一个 URL 就会在测试中真正访问外部服务。使用 `@respx.mock(assert_all_mocked=True)` 或 router 化 `respx.mock(assert_all_mocked=True, assert_all_called=True)`，未匹配请求直接抛错，保证测试完全离线。

### 5.5 时间：freezegun

涉及时间敏感逻辑（token 过期、锁定 TTL、任务状态）用 `freeze_time`：

```python
from freezegun import freeze_time

@freeze_time("2026-01-01 10:00:00")
async def test_token_expired(db, mock_redis):
    ...
```

> **DB 端默认时间不受 freeze 控制**：SQLAlchemy 模型若用 `server_default=func.now()`（数据库端生成时间），freezegun 只冻结 Python 时间，insert 后从 DB 读回的时间仍是真实时间。本项目模型统一用 Python 端 `default=lambda: datetime.now()`，freezegun 生效；改造时不要引入 `server_default` 时间字段，否则时间敏感测试需改用 SQL 注入或查询时手动覆盖。

### 5.6 仓库层依赖：构造注入 + AsyncMock

service 层仓库依赖应**构造注入**（替换模块级单例 + 方法内延迟导入），测试时注入 mock 仓库即可，无需 monkeypatch 模块属性：

```python
# app/service/xxx_service.py（目标形态）
class QuotaService:
    def __init__(self, member_repository=None, member_benefit_repository=None):
        self.member_repository = member_repository or default_member_repository
        self.member_benefit_repository = member_benefit_repository or default_member_benefit_repository
```

```python
# 测试
async def test_quota(db, mock_redis, mocker):
    svc = QuotaService(
        member_repository=mocker.AsyncMock(return_value=member),
        member_benefit_repository=mocker.AsyncMock(return_value=benefit),
    )
    result = await svc.get_quota(user_id=1)
    assert result.daily_left == expected  # 断言业务结果
```

### 5.7 Elasticsearch：respx 拦截

elasticsearch-py 8.x 的 async 客户端底层走 httpx，直接用 respx 按 URL 拦截，替代手写 fake client / monkeypatch `_client` 属性：

```python
import respx
from httpx import Response

@respx.mock(assert_all_mocked=True)
async def test_vector_search():
    respx.post(f"{settings.ES_URL}/memories/_search").mock(
        return_value=Response(200, json={"hits": {"hits": [...]}})
    )
    hits = await es_service.vector_search(...)
    assert hits == [...]
```

> ES 地址从 `app.config.settings` 读取（如 `settings.ES_URL`），不要硬编码 `http://es:9200`——与真实环境/CI 配置解耦。

### 5.8 RabbitMQ：AsyncMock 替换连接

aio-pika 无成熟内存模拟库。测试重点放在 handler 业务逻辑（直接调用 handler 函数、patch 其依赖）和 Publisher/Consumer 调用契约（patch `aio_pika.connect_robust` 返回假连接）：

```python
async def test_export_handler(mocker):
    mocker.patch("aio_pika.connect_robust", return_value=mocker.AsyncMock())
    # 直接调用 consumer handler，patch 其 repository 依赖
    await handle_export_task(message_payload, db=db, storage=local_storage)
    # 断言任务状态/导出结果，而非断言 MQ 调用序列
```

### 5.9 MinIO：依赖抽象注入，不 mock 客户端

service 层应依赖 `StorageService` 抽象（`app/service/storage/base.py`），测试直接注入 `LocalStorage` 或内存 fake，不需要 mock Minio 客户端：

```python
@pytest.fixture
def storage(tmp_path):
    return LocalStorage(base_dir=str(tmp_path))  # 真实文件读写，无需 mock
```

仅在验证 `minio_storage.py` 本身时，才 mock `get_minio_client` 单例（AsyncMock 返回预设对象）。

## 6. Fixture 规范

- **分层**：跨模块共享的基础 fixture（`db`、`mock_redis`、`mongo_db`）放 `tests/conftest.py`；仅单模块使用的 fixture 放该测试文件内。
- **职责单一**：fixture 只负责装配外部依赖，业务数据构造用 factory 函数/类，不要塞进 fixture。
- **命名**：fixture 用蛇形命名（`mock_redis`、`mongo_db`、`db`），与依赖名同义，不追加无意义后缀。
- **数据生成**：实体构造用 `faker` + 模块内工厂函数（如 `user_factory(db, **overrides)`），避免每个测试手写 ORM 实体。
- **拆分阈值（演进指引）**：根 `conftest.py` 超过约 100 行或 6 个以上 fixture 时，按域下沉到子目录 conftest（如 `tests/service/<module>/conftest.py`）；仅当多个目录共享同一 fixture 时才考虑 `tests/fixtures/` 包 + `pytest_plugins` 显式声明。当前规模（2 个 fixture、53 行）不拆分，避免过度设计。
- **配置覆盖**：应用配置以 `app/config.py`（pydantic-settings）为单一来源，测试**不新建独立 `config.py`**。按需用 `monkeypatch.setattr(settings, "KEY", value)` 覆盖单条配置；仅当同一覆盖在 10+ 处重复时才收敛为 `settings_override(**kw)` 之类的小 fixture。
- **数据确定性**：`faker` 生成数据用固定 seed（`Faker(seed=0)` 或 `faker.seed_instance(0)`）保证可复现，随机数据会导致偶发失败且难排查。时间类断言用 freezegun 冻结（见 5.5），不要用真实时间。
- **作用域**：fixture 默认 `function` 作用域（每测试独立，数据隔离靠重建而非清理）；需要跨测试共享的只读资源才用 `session` 作用域，且必须是纯只读（如不变化的常量工厂）。

## 7. 断言与命名规范

- 断言业务结果/状态变化，不断言 mock 调用序列。
- 期望抛异常用 `pytest.raises(BusinessException, match="关键信息")`，`match` 写业务语义而非错误码全称。
- 测试函数名：`test_<方法>_<场景>_<期望>`，如 `test_login_success_creates_session`、`test_quota_exceeded_raises_business_error`。
- 分组：同一 service 的测试可放同一文件顶层函数式书写（本项目风格），用 `async def test_*`，依赖 `asyncio_mode = auto`，无需 `@pytest.mark.asyncio`。

## 8. 禁止项（反面清单）

| 反模式 | 替代 |
|--------|------|
| `run_coro(coro)` 手动建 event loop | 直接 `async def test_*`（asyncio_mode=auto） |
| `async_ret(value)` 手写 async 桩 | `mocker.AsyncMock(return_value=value)` |
| `StubAsyncSession`/`NullDBSession` 手写 Session 桩 | `db` fixture（SQLite 内存库） |
| 手写 HTTP 假件（`FakeStreamResponse` 等） | respx |
| 手写 ES fake client / monkeypatch `_client` | respx 拦截 ES HTTP 请求 |
| 手写 Mongo dict 假件（`mongomock.MongoClient` 同步版） | `mongomock_motor.AsyncMongoMockClient` |
| 为 MinIO 引入专门 mock 库 | 依赖 `StorageService` 抽象，注入 `LocalStorage` |
| monkeypatch 模块级单例仓库属性 | 构造注入 + AsyncMock |
| 空文件 / 无引用的死桩 | 删除（须先核实完整引用链，含内部引用：`FakeGraph` 曾被 `install_reasoning_chain_mocks` 内部使用，只看 import 语句会误判为死代码） |
| `test_xxx_gaps`/`test_xxx_repro` 补丁式文件 | 整理进原测试文件 |
| `SimpleNamespace` 冒充 ORM 实体作为结果断言 | 真实 SQLite 写入后查询断言 |
| router 测试 override `get_db` 返回 `object()` | 返回 `db` fixture 的真实 session（见 4.1） |

## 8.5 Marker 落地规则

`pytest.ini` 已声明 6 个 marker，使用规则：

| marker | 适用对象 | 何时打 |
|--------|---------|--------|
| `unit` | 纯逻辑/service 单测（无外部依赖） | 默认即单测，可省略 |
| `slow` | 执行时间 > 2s 的测试（如大文件处理） | 打上便于 CI 分层跳过 |
| `integration` | 依赖真实中间件（testcontainers） | 未来集成测试 |
| `api` | router 契约测试 | 打到 router 测试文件 |
| `requires_db` / `requires_redis` | 需要真实中间件的测试 | 无内存替身时才打，通常不需要 |

默认 `pytest` 跑全部；CI 中慢速/集成用 `-m "not slow and not integration"` 分层。**不要为上述规则之外的目的发明新 marker**（`--strict-markers` 会拒绝未声明 marker）。

## 8.6 覆盖率目标

- 目标：核心 service/repository 模块 ≥ 80%，整体 ≥ 60%（当前无门槛，逐步收敛）。
- 查看方式：`uv run pytest --cov=app --cov-report=term-missing`。
- 覆盖率是手段不是目的：优先保证行为断言正确，再追求覆盖；不写"为覆盖而覆盖"的无效断言。

## 9. 运行方式

```bash
# 安装/更新依赖（test extra 含全部测试库）
uv sync --extra test

# 运行全部测试
uv run pytest

# 单模块 / 单文件
uv run pytest tests/service/test_auth_service.py

# 覆盖率
uv run pytest --cov=app --cov-report=term-missing
```

## 10. 现状差距与改造优先级

| 优先级 | 改造项 | 说明 |
|--------|--------|------|
| P0 | `async_ret` → `AsyncMock`、`run_coro` → async 测试 | 机械替换，不改断言语义 |
| P1 | `conftest.py` 增加 `db`（SQLite 内存库）fixture | 需先验证建表兼容性 |
| P1 | `StubAsyncSession` 系列替换为 `db` fixture | 逐模块推进 |
| P1 | `conftest.py` 增加 `mongo_db`（mongomock-motor）fixture | 覆盖登录日志/审计日志模块 |
| P1 | billing/ai 模块 repository 构造注入 | 消除模块单例 monkeypatch |
| P2 | `install_reasoning_chain_mocks` god-mock 收敛 | 只替身外部世界：真实 SQLite + 保留 graph.astream 等边界 mock，`_finalize_message`/`_push_end` 等内部方法走真实逻辑，断言业务结果而非调用序列（依赖 P1 的 `db` fixture 落地） |
| P2 | 手写 HTTP 假件替换为 respx | 配合 respx 接入 |
| P2 | ES / MQ / MinIO 模块测试按 §5.7-5.9 规范改造 | 复用 respx / AsyncMock / 抽象注入 |
| P2 | 按 §8.5 落地 marker 标注 | 与 pytest.ini 对齐，slow/api 等按规则补标 |
| P3 | `stubs.py` 按职责拆分（fakes/factories/mocks） | 结构优化 |

## 11. 与源码架构改造的协同策略

tests 改造（§10 P0-P3）与源码服务层架构治理（[Python 后端改造计划](../../dehaze-doc/docs/05-改造计划/Python后端改造计划.md) §6：`ai/` 分子包 → infrastructure 下沉 → god-service 拆分）同期推进。本节给出配合方式，目标是**消灭两类无用功**：

1. 源码 Move 后手工改测试 import——PyCharm 免费完成的事，人工做就是重复劳动
2. 按旧结构深度重写测试，源码随后拆分——重写作废

### 11.1 策略一：import 路径零手工改动

- tests 与 app 在同一个 PyCharm 项目内。源码批次 1/2 的 `Refactor → Move` 会**自动同步 tests 下全部 import**：已勘察 tests 中 37 个文件引用 `app.service.ai.` / `app.infrastructure.llm.` 路径，一次 Move 全覆盖（含 embedding/rerank 等跨模态引用改写为 `provider/`）
- **禁止**在源码 Move 前用 sed / 全局替换手工改测试 import——路径被提前改掉后，PyCharm Move 无法匹配旧路径，自动同步失效，反而制造双倍劳动
- 当前无 `importlib`/`__import__` 动态加载（已勘察），不存在 Move 覆盖不到的引用

### 11.2 策略二：tests 目录结构滞后对齐（先搬源码，后搬测试）

- tests 目录只是组织性容器：pytest 按文件收集，测试内部统一 `from app.xxx` 引用被测模块，**tests 自身位置不影响任何行为**
- 因此源码批次 1/2 完成后，`tests/service/ai/` 55 个测试文件保持平铺不动，仅 import 由 IDE 自动更新；等源码最终结构稳定（批次 3 完成后）再一次性按 §3 规范镜像对齐：`tests/service/ai/service|builders|middleware|strategies/`、下沉文件对应测试移入 `tests/infrastructure/` 对应目录
- 收益：源码每批只搬一次、tests 只搬一次（终态对齐），避免"源码动一次、tests 跟一次"的双倍搬移
- **例外：新增测试按目标结构直接落位**——如批次 2 后新增 a2a 测试放 `tests/infrastructure/a2a/`、新增 provider 测试放 `tests/infrastructure/provider/`，不放入将废弃的位置，避免二次搬移

### 11.3 策略三：内容改造（§10 P0-P3）与源码批次并行但有序

| 测试改造项 | 与源码批次的配合方式 |
|------------|----------------------|
| P0 `async_ret`/`run_coro` 机械替换 | 与源码 Move 完全正交，**优先做**（批 1 前完成，后续改造基线干净） |
| P1 `db`/`mongo_db` fixture、构造注入 | 按模块独立推进，与搬移不冲突；但**批次 3 涉及拆分的 god-service 测试**（`test_task_service_consume`、`test_prediction_batch_validation`、`test_order_permission_fix` 等）先只做 P0，等源码拆分完成后再做 P1/P2——避免按旧模块重写后又要重组 |
| P2 god-mock 收敛（reasoning 链） | `reasoning_service` 批次 1 仅搬位置不改内容，不冲突；依赖 P1 的 `db` fixture 落地，须排在 P1 之后 |
| P2 respx / ES / MQ / MinIO | 与结构改造正交，随时可做 |
| P3 `stubs.py` 拆分 | 纯结构优化，任意时间 |

总原则：**结构跟随源码，内容按模块独立推进；深度重构（god-mock 收敛、测试拆分重组）永远等该模块源码稳定后做**。

### 11.4 策略四：每批验收闸门

- 源码批次 1/2 完成后执行 `grep -rn "from app\.service\.ai\." tests/ app/` 应为空（Move 漏网检测；批次 2 另加 `grep -rn "llm\.a2a_\|llm\.provider_"`）
- 全量 `uv run pytest` 必须绿；若失败源于 import，属 Move 漏引用，**当场修复，不留到下一批**
- 涉及批次 3 的 god-service 测试拆分须与源码拆分同一批完成，测试全绿后该批才算收口
