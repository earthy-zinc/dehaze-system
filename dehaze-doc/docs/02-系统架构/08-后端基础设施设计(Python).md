# 后端基础设施设计（dehaze-python）

## 1. 文档概述

### 1.1 文档目的

本文档描述 `dehaze-python` 后端项目的基础设施层设计，包括项目分层架构、应用生命周期、配置管理、数据访问层、缓存体系、消息队列、定时任务、安全中间件、日志系统和可观测性等基础能力。

本文档**不涉及**具体业务模块的实现逻辑和去雾算法模块，业务模块详见 [模块设计](../03-模块设计/) 各子目录。

### 1.2 适用范围

面向参与 `dehaze-python` 后端开发的工程师，提供技术基座的全局视图和设计决策依据。

### 1.3 相关文档

| 文档 | 说明 |
|------|------|
| [总体架构设计](./01-总体架构设计.md) | 系统全局分层、数据流与安全策略 |
| [数据库设计](./03-数据库设计.md) | 表结构、索引、ER 关系图 |
| [API 规范](./04-API规范.md) | 全局 API 规范、认证方式、错误码 |
| [后端基础设施设计(Java)](./09-后端基础设施设计(Java).md) | Java 端对等基础设施设计 |
| [后端基础设施设计(Go)](./07-后端基础设施设计(Go).md) | Go 端对等基础设施设计 |

---

## 2. 项目目录结构

```text
dehaze-python/
├── app/                               # 主应用目录
│   ├── __init__.py                    # 延迟导入入口（app / settings）
│   ├── main.py                        # FastAPI 应用入口（Lifespan + 路由注册 + 健康检查）
│   ├── config.py                      # 多环境配置（Pydantic Settings）
│   ├── database.py                    # SQLAlchemy 2.0 异步引擎 & Session 工厂
│   ├── dependencies/                  # FastAPI 依赖注入
│   │   ├── auth.py                    # JWT 认证依赖（UserContext / get_current_user）
│   │   ├── common.py                  # Redis 连接池 & 健康检查
│   │   └── permission.py             # 权限校验依赖工厂（PermissionChecker）
│   ├── decorators/                    # 横切关注点装饰器
│   │   └── permission.py             # 权限检查装饰器（require_permission / require_any_permission）
│   ├── middleware/                     # ASGI 中间件
│   │   └── operation_log.py          # 操作日志中间件（异步写入 MySQL）
│   ├── metrics/                       # Prometheus 指标采集
│   │   ├── gpu_metrics.py            # GPU 利用率 / 显存 / 温度
│   │   ├── http_metrics.py           # HTTP 请求量 / 延迟
│   │   ├── inference_metrics.py      # 推理耗时 / 请求计数
│   │   └── task_metrics.py           # 任务队列深度 / 处理耗时
│   ├── mq/                            # RabbitMQ 消息队列
│   │   ├── publisher.py              # 消息发布（自动重连 + 指数退避）
│   │   ├── consumer.py               # 多队列消费（handler 注册 + ack/nack）
│   │   ├── connection.py             # 全局单例管理 + Lifespan 集成
│   │   └── handlers.py               # 消费者 handler（导出任务等）
│   ├── job/                           # XXL-Job 定时任务
│   │   ├── executor.py               # 执行器生命周期管理（pyxxl daemon）
│   │   └── handlers.py               # 定时任务 handler（清理/回收/健康检查）
│   ├── models/                        # 数据模型
│   │   ├── base.py                    # BaseModel（SQLAlchemy 事件自动填充审计字段）
│   │   ├── entity/                    # ORM 实体（与表对应）
│   │   ├── schema/                    # Pydantic Schema（API 请求/响应）
│   │   ├── form/                      # 服务层内部表单对象
│   │   ├── vo/                        # 视图对象（复合响应结构）
│   │   └── enum/                      # 枚举常量
│   ├── repository/                    # Repository 层（数据访问抽象）
│   │   ├── base.py                    # 泛型 BaseRepository（CRUD / 分页 / 模糊搜索）
│   │   ├── user_repository.py        # 用户数据访问
│   │   ├── dataset_repository.py     # 数据集数据访问
│   │   └── ...                        # 其他 Repository
│   ├── route/                         # 路由层（Controller）
│   │   ├── __init__.py               # 统一路由注册
│   │   ├── auth.py                    # 认证路由
│   │   ├── websocket.py              # WebSocket 路由
│   │   └── ...                        # 其他业务路由
│   ├── service/                       # 服务层（业务逻辑）
│   │   ├── task_tracker.py           # 任务追踪管理器（优雅关闭支持）
│   │   ├── websocket_service.py      # WebSocket 连接管理
│   │   └── ...                        # 其他业务 Service
│   └── utils/                         # 工具层
│       ├── code.py                    # ResultCode 统一错误码
│       ├── result.py                  # 统一响应封装（Result 泛型）
│       ├── exceptions.py             # BusinessException + 全局异常处理器
│       ├── jwt_utils.py              # JWT 工具
│       ├── logging.py                # 日志系统（UTF-8 轮转处理器）
│       ├── redis_fallback.py         # Redis 降级 / 重试 / 熔断器
│       ├── metrics.py                # 图像质量指标（PSNR/SSIM/LPIPS）
│       └── ...                        # 其他工具
├── algorithm/                         # 去雾算法模块（34 种算法）
├── config.py                          # 算法模块配置（设备 / 路径）
├── migrations/                        # Alembic 数据库迁移
├── tests/                             # 测试
│   └── conftest.py                   # pytest fixtures
├── pyproject.toml                     # 项目依赖（uv 管理）
├── Dockerfile                         # GPU 推理容器化
└── logs/                              # 运行时日志目录
```

---

## 3. 分层架构设计

### 3.1 架构分层

项目采用 **FastAPI Lifespan + 异步四层架构 + 依赖注入** 的设计。

```mermaid
flowchart TB
    subgraph External["外部请求"]
        Client["HTTP Client"]
        WS["WebSocket Client"]
    end

    subgraph Middleware["中间件 / 依赖注入链"]
        direction LR
        CORS["CORS 跨域"]
        OpLog["操作日志<br/>OperationLogMiddleware"]
        JWT["JWT 认证<br/>get_current_user"]
        Permission["权限校验<br/>PermissionChecker"]
    end

    subgraph Route["Route 层 (app/route/)"]
        direction LR
        R1["参数绑定 (Pydantic)"]
        R2["参数校验"]
        R3["响应封装"]
    end

    subgraph Service["Service 层 (app/service/)"]
        direction LR
        S1["业务编排"]
        S2["事务管理"]
        S3["缓存交互"]
    end

    subgraph Repository["Repository 层 (app/repository/)"]
        direction LR
        Repo1["BaseRepository<br/>泛型 CRUD"]
        Repo2["子类扩展<br/>特定查询"]
    end

    subgraph Models["Models 层 (app/models/)"]
        direction LR
        Entity["Entity (ORM)"]
        Schema["Schema (Pydantic)"]
        VO["VO (视图对象)"]
    end

    subgraph Infra["基础设施层"]
        direction LR
        MySQL[("MySQL")]
        Redis[("Redis")]
        MinIO[("MinIO")]
        MQ[("RabbitMQ<br/>(规划)")]
        Kafka[("Kafka<br/>(规划)")]
    end

    Client --> Middleware --> Route --> Service --> Repository --> Infra
    Service --> Models
    WS -->|"原生 WebSocket"| Service
```

### 3.2 层级职责

| 层级 | 包路径 | 职责 | 依赖方向 |
|------|--------|------|----------|
| **中间件/依赖注入** | `middleware/` + `dependencies/` | 请求拦截、认证、鉴权、操作日志 | ← 外部请求 |
| **Route 层** | `route/` | 参数绑定与校验（Pydantic）、调用 Service、统一响应 | → Service |
| **Service 层** | `service/` | 业务逻辑编排、事务边界、缓存交互、异步任务 | → Repository + 基础设施 |
| **Repository 层** | `repository/` | 数据库 CRUD 封装、分页、模糊搜索、复杂查询 | → SQLAlchemy ORM |
| **Models 层** | `models/` | ORM 实体、Schema 定义、VO/Form/Enum | 被 Route / Service / Repository 依赖 |
| **基础设施层** | `database.py` + `dependencies/` + `utils/` | 数据库、缓存、对象存储、日志等基础能力 | 被所有层依赖 |

### 3.3 依赖注入策略

项目基于 **FastAPI 原生依赖注入系统**实现：

- 使用 `Depends()` 声明依赖关系，由框架自动解析
- 数据库 Session 通过 `get_db` 异步生成器注入，自动管理生命周期
- Redis 连接通过 `get_redis` 生成器注入，或通过 `get_redis_client()` 获取全局单例
- 权限校验通过 `PermissionChecker` 工厂函数生成 `Depends` 包装器

```mermaid
flowchart LR
    Route["Route Handler"] --> Depends["Depends(get_current_user)"]
    Depends --> JWT["JWT 解析 → UserContext"]
    Route --> Depends2["Depends(get_db)"]
    Depends2 --> Session["AsyncSession"]
    Route --> Depends3["PermissionChecker('sys:user:add')"]
    Depends3 --> Check["权限校验"]
```

### 3.4 数据模型分层

```mermaid
flowchart LR
    Request["HTTP Request"] --> Schema["Schema (Pydantic)<br/>API 文档 + 校验"]
    Schema --> Entity["Entity (SQLAlchemy)<br/>数据库表映射"]
    Entity --> VO["VO (视图对象)<br/>API 响应"]

    Form["Form (内部对象)<br/>服务间传递"] -.-> Service["Service 层"]
    Enum["Enum (枚举)<br/>常量定义"] -.-> Service
```

| 模型类型 | 包路径 | 职责 | 示例 |
|----------|--------|------|------|
| **Entity** | `models/entity/` | 数据库表映射，SQLAlchemy Column 定义 | `SysUser` |
| **Schema** | `models/schema/` | API 请求/响应 Schema，Pydantic 校验 + OpenAPI 自动生成 | `UserPageQuery` |
| **VO** | `models/vo/` | 复合视图对象，多表联查结果封装 | `DatasetVO` |
| **Form** | `models/form/` | 服务层内部数据传输对象 | `DatasetAddForm` |
| **Enum** | `models/enum/` | 枚举常量定义 | `TaskStatus` |

---

## 4. 应用生命周期管理

### 4.1 启动流程

基于 FastAPI Lifespan 上下文管理器实现完整的生命周期控制：

```mermaid
sequenceDiagram
    participant Uvicorn as uvicorn
    participant Lifespan as lifespan()
    participant Config as 配置加载
    participant DB as 数据库初始化
    participant Redis as Redis 初始化
    participant Tracker as 任务追踪器
    participant Metrics as Prometheus 指标
    participant Route as 路由注册

    Uvicorn->>Lifespan: 启动 ASGI 应用
    Lifespan->>Config: 加载环境配置<br/>(Pydantic Settings)
    Note right of Config: DevelopmentSettings<br/>TestingSettings<br/>ProductionSettings
    Lifespan->>DB: init_db()<br/>异步引擎 + 连接测试
    Lifespan->>Redis: get_redis_client()<br/>连接池 + 健康检查
    Lifespan->>Tracker: init_task_tracker()<br/>后台任务追踪
    Lifespan->>Metrics: collect_gpu_metrics()<br/>(条件启用)
    Note right of Route: 10 个 APIRouter<br/>+ WebSocket 路由
    Lifespan-->>Uvicorn: yield（应用就绪）
```

### 4.2 优雅关闭

uvicorn 收到 `SIGTERM/SIGINT` 后自动触发 Lifespan 关闭阶段，无需自定义信号处理器：

```text
收到 SIGINT/SIGTERM
    → 任务追踪器进入关闭模式（拒绝新任务）
    → 通知 WebSocket 客户端断开
    → 等待运行中任务完成（超时 30s，超时后取消）
    → 停止 GPU 指标采集器
    → 关闭 Redis 连接池
    → 关闭数据库连接池（engine.dispose）
```

### 4.3 部署模式

| 环境 | 启动方式 | 并发模型 |
|------|----------|----------|
| **开发** | `uvicorn app.main:app --reload` | 单 Worker + 热重载 |
| **生产** | `uvicorn app.main:app --host 0.0.0.0 --port 80 --workers 4` | 多 Worker 进程 |
| **Docker** | NVIDIA CUDA 12.1 基础镜像 | uvicorn 多 Worker + GPU 推理 |

> **注意**：Dockerfile 当前仍引用旧的 Flask/Gunicorn 方式，需更新为 uvicorn 启动命令。

---

## 5. 配置管理

### 5.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 配置框架 | Pydantic Settings v2 | 类型安全、自动校验、环境变量绑定 |
| 环境变量 | `.env` 文件 + `os.getenv` | 敏感信息外部化 |
| 配置切换 | `APP_ENV` 环境变量 | 决定加载哪个配置类 |
| 计算属性 | `@computed_field` | 自动派生 DATABASE_URL / REDIS_URL |

### 5.2 配置结构

```python
class Settings(BaseSettings):
    # 应用基础
    APP_NAME / APP_VERSION / DEBUG

    # JWT 配置（启动时强制校验非空）
    SECRET_KEY / JWT_SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES       # 访问令牌 2h
    JWT_REFRESH_TOKEN_EXPIRES      # 刷新令牌 7d

    # 数据库配置
    DB_HOST / DB_PORT / DB_NAME / DB_USER
    DATABASE_POOL_SIZE / DATABASE_MAX_OVERFLOW / DATABASE_POOL_RECYCLE
    DATABASE_URL                   # @computed_field 自动拼接

    # Redis 配置
    REDIS_HOST / REDIS_PORT / REDIS_DB
    REDIS_MAX_CONNECTIONS / REDIS_SOCKET_TIMEOUT / REDIS_HEALTH_CHECK_INTERVAL
    REDIS_URL                      # @computed_field 自动拼接

    # MinIO 对象存储配置
    MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_BUCKET_NAME

    # Prometheus 监控
    PROMETHEUS_ENABLED / PROMETHEUS_GPU_COLLECT_INTERVAL

    # 优雅关闭
    GRACEFUL_SHUTDOWN_TIMEOUT / GRACEFUL_SHUTDOWN_CANCEL_ON_TIMEOUT

    # 设备配置
    DEVICE_ID                      # GPU 设备 ID 列表
```

### 5.3 多环境支持

| 环境 | 配置类 | `APP_ENV` | 特性差异 |
|------|--------|-----------|----------|
| **开发** | `DevelopmentSettings` | `development` | DEBUG=True，SQL 日志输出，远程开发数据库 |
| **测试** | `TestingSettings` | `testing` | DEBUG=True，独立测试数据库 `dehaze_test` |
| **生产** | `ProductionSettings` | `production` | 强制校验密钥长度 ≥ 32 且 `DEHAZE_PASSWORD` 非空 |

### 5.4 敏感信息管理

所有敏感配置通过 `.env` 文件或环境变量注入：

```bash
# .env
SECRET_KEY=your-secret-key-at-least-32-chars
JWT_SECRET_KEY=your-jwt-secret-key
DEHAZE_PASSWORD=shared-password-for-mysql-redis-minio
```

**安全校验**：
- `Settings.__init__` 在所有环境校验 `SECRET_KEY` 和 `JWT_SECRET_KEY` 非空
- `ProductionSettings.__init__` 额外校验密钥长度 ≥ 32 且 `DEHAZE_PASSWORD` 非空

---

## 6. 数据访问层

### 6.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| ORM | SQLAlchemy 2.0 + 异步模式 | 声明式模型、AsyncSession |
| 数据库驱动 | aiomysql（异步）+ PyMySQL（同步） | 异步为主，同步用于 Alembic 迁移 |
| 数据库迁移 | Alembic (纯 CLI) | 版本化 Schema 管理 |
| 对象存储 | MinIO Python SDK | 文件/图像存储 |

### 6.2 异步引擎 & Session 管理

```python
# database.py 核心组件
engine = create_async_engine(settings.DATABASE_URL, ...)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, ...)

# FastAPI 依赖注入
async def get_db() -> AsyncGenerator[AsyncSession, None]: ...

# 非依赖注入场景（后台任务等）
async with get_db_session() as session: ...
```

连接池参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_size` | 10 | 连接池常驻连接数 |
| `max_overflow` | 20 | 最大溢出连接数 |
| `pool_recycle` | 3600s | 连接回收周期 |
| `echo` | False（开发 True） | SQL 日志输出 |

#### 事务管理策略

采用 **"请求边界 = 事务边界"** 模型，由 `get_db()` / `get_db_session()` 统一管理 commit/rollback，Service/Repository 层无需手动提交：

```python
# get_db() — FastAPI 路由依赖注入
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()       # 请求正常完成 → 自动提交
        except Exception:
            await session.rollback()     # 异常抛出 → 自动回滚
            raise

# get_db_session() — 非 HTTP 场景（后台任务、消息消费者）
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

各层职责划分：

| 层级 | 事务职责 | 说明 |
|------|----------|------|
| Route 层 | 事务边界持有者 | 通过 `Depends(get_db)` 获取 session，请求结束自动 commit/rollback |
| Service 层 | 业务编排 | 组合多个 Repository 操作，不做 commit |
| Repository 层 | 数据操作 | 只做 `flush()`（获取自增 ID 时），不做 commit |
| 后台任务 | 自管事务 | 使用 `async_session_factory()` 直接创建 session，按需显式 commit（长任务需中间持久化状态） |

> **设计决策**：不引入 `@transactional` 装饰器。FastAPI 的 `Depends(get_db)` 天然就是事务边界的最佳承载点，装饰器会与 DI 签名冲突、且需要额外实现事务传播语义（REQUIRED/REQUIRES_NEW 等），性价比不高。

### 6.3 Repository 层

引入泛型 `BaseRepository[T]`，提供标准 CRUD 操作，子类只需声明 `model` 类型即可继承全部能力：

```mermaid
classDiagram
    class BaseRepository~T~ {
        +model: type[T]
        +get_by_id(db, id) T?
        +get_all(db) list[T]
        +paginate(db, stmt, page, size) tuple
        +apply_keyword_filter(stmt, columns, keyword) Select
        +create(db, entity) T
        +create_all(db, entities) list[T]
        +save(db, entity) T
        +delete_by_ids(db, ids) int
        +soft_delete_by_ids(db, ids) int
        +count(db, stmt?) int
        +exists_by_id(db, id) bool
    }

    class UserRepository {
        +model = SysUser
        +get_by_username(db, username) SysUser?
        +get_user_page(db, query) tuple
    }

    class DatasetRepository {
        +model = SysDataset
        +get_dataset_with_items(db, id) DatasetVO?
    }

    BaseRepository <|-- UserRepository
    BaseRepository <|-- DatasetRepository
```

**设计决策**：Repository 层将 ORM 查询逻辑从 Service 层剥离，降低耦合度，便于单元测试时 Mock 数据层。

### 6.4 BaseModel 审计字段自动填充

通过 SQLAlchemy `event.listens_for` 事件机制 + `ContextVar` 实现：

| 事件 | 触发时机 | 自动填充字段 |
|------|----------|-------------|
| `before_insert` | INSERT 前 | `create_time`、`update_time`、`create_by`、`update_by` |
| `before_update` | UPDATE 前 | `update_time`、`update_by` |

当前用户 ID 通过 `ContextVar` 线程安全地传递：

```text
get_current_user (依赖注入) → set_current_user_id(user.id)
    → ContextVar 存储
    → before_insert/before_update 事件读取
```

### 6.5 存储分工

```mermaid
flowchart LR
    subgraph Relational["MySQL (结构化数据)"]
        User["用户/角色/部门"]
        Business["数据集/算法/任务"]
        Log["操作日志"]
    end

    subgraph Object["MinIO (文件存储)"]
        Image["原始图像/去雾结果"]
        Model["模型权重文件"]
        Export["导出文件"]
    end
```

---

## 7. 缓存体系

### 7.1 架构设计

当前使用 Redis 异步客户端作为唯一缓存层，通过连接池管理连接生命周期：

```mermaid
flowchart TB
    subgraph CacheArch["缓存体系"]
        direction TB
        subgraph DI["依赖注入方式"]
            GetRedis["get_redis()<br/>生成器，自动关闭"]
        end
        subgraph Singleton["全局单例方式"]
            GetClient["get_redis_client()<br/>后台任务 / 中间件"]
        end
        subgraph Resilience["弹性机制"]
            Fallback["redis_operation_with_fallback<br/>优雅降级"]
            Retry["@with_redis_retry<br/>自动重试"]
            CB["RedisCircuitBreaker<br/>熔断器"]
        end
    end

    subgraph Backend["Redis (异步)"]
        Pool[("ConnectionPool<br/>max_connections=20")]
    end

    DI --> Pool
    Singleton --> Pool
    Resilience --> Pool
```

### 7.2 连接池配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_connections` | 20 | 最大连接数 |
| `socket_timeout` | 5.0s | 操作超时 |
| `socket_connect_timeout` | 5.0s | 连接超时 |
| `retry_on_timeout` | True | 超时是否重试 |
| `health_check_interval` | 30s | 健康检查间隔 |

### 7.3 缓存用途矩阵

| 用途 | Key 格式 | TTL | 说明 |
|------|----------|-----|------|
| 验证码 | `captcha:{key}` | 5min | 登录验证码 |
| Token 黑名单 | `token:blacklist:{jti}` | Token 剩余有效期 | JWT 注销 |
| 任务状态 | `export:task:{task_id}` | 24h | 导出任务进度缓存 |
| 任务取消标志 | `task:cancel:{task_id}` | 5min | 标记任务取消 |

### 7.4 Redis 弹性机制

| 机制 | 实现 | 说明 |
|------|------|------|
| **优雅降级** | `redis_operation_with_fallback()` | Redis 不可用时执行 fallback 函数或返回默认值 |
| **自动重试** | `@with_redis_retry` 装饰器 | 指数退避重试，默认 3 次 |
| **熔断器** | `RedisCircuitBreaker` | 连续 5 次失败后熔断，30s 后半开尝试恢复 |

```mermaid
stateDiagram-v2
    [*] --> CLOSED
    CLOSED --> OPEN: 连续失败 ≥ 5 次
    OPEN --> HALF_OPEN: 冷却 30s 后
    HALF_OPEN --> CLOSED: 请求成功
    HALF_OPEN --> OPEN: 请求失败
```

### 7.5 缓存演进规划

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | 缓存 Key 统一管理 | 引入 `CacheKey` 枚举类，避免 Key 散落各处 |
| P2 | 推理结果缓存 | 相同输入图像 + 相同算法的推理结果可缓存，避免重复计算 |
| P3 | 与 Go/Java 端对齐多级缓存 | 评估引入本地缓存层（Python 端受限于 GPU 内存优先级较低） |

---

## 8. 消息队列

### 8.1 技术选型

与 Java/Go 端保持一致的中间件选型：

| 消息中间件 | 用途 | 当前状态 |
|------------|------|----------|
| **RabbitMQ** | 异步任务分发（导出、批量操作等） | ✅ 已实现（aio-pika + MQ 优先 / asyncio.Task fallback） |
| **Kafka** | 日志收集与流处理 | 📋 规划中 |

### 8.2 双通道任务分发架构

采用 **MQ 优先 + asyncio.Task fallback** 双通道架构，RabbitMQ 可用时消息持久化投递，不可用时自动降级为进程内异步任务：

```mermaid
flowchart TB
    subgraph Dispatch["任务分发 (_dispatch_task)"]
        Create["创建任务<br/>写入 DB + Redis"]
        Check{"RabbitMQ<br/>可用?"}
    end

    subgraph MQPath["MQ 通道 (持久化)"]
        Publish["Publisher.publish()<br/>routing_key=task.export"]
        Queue["task.export 队列<br/>(durable, TTL=24h)"]
        Handler["handle_export_task<br/>Consumer 回调"]
    end

    subgraph FallbackPath["Fallback 通道 (进程内)"]
        AsyncTask["asyncio.create_task()"]
        Tracker["TaskTracker 注册"]
    end

    subgraph Execute["任务执行"]
        Background["_execute_export_task_background"]
    end

    Create --> Check
    Check -->|"是"| Publish --> Queue --> Handler --> Background
    Check -->|"否 / 发布失败"| AsyncTask --> Tracker --> Background
```

**双通道对比**：

| 特性 | MQ 通道 (RabbitMQ) | Fallback 通道 (asyncio.Task) |
|------|-------------------|-----------------------------|
| 消息持久化 | ✅ PERSISTENT 模式 | ❌ 进程内存 |
| 进程崩溃恢复 | ✅ 消息留在队列 | ❌ 任务丢失 |
| ACK 确认 | ✅ 处理成功后 ack | ❌ 无 |
| 失败重试 | ✅ nack + requeue | ❌ 需手动实现 |
| 死信队列 | ✅ DLX 兜底 | ❌ 无 |
| 跨进程/跨语言 | ✅ 共享队列 | ❌ 仅进程内 |

### 8.3 Fallback 方案：asyncio.Task + TaskTracker

RabbitMQ 不可用时的降级方案，通过 `TaskTracker` 追踪进程内异步任务：

**TaskTracker 核心能力**：

| 能力 | 说明 |
|------|------|
| 任务注册 | `register(task_id, task, task_type, metadata)` |
| 关闭模式 | `initiate_shutdown()` 后拒绝新任务注册 |
| 等待完成 | `wait_for_completion(timeout)` 等待所有任务，超时后取消 |
| 自动清理 | 任务完成时通过 `add_done_callback` 自动移除 |

### 8.4 任务状态流转

```mermaid
stateDiagram-v2
    [*] --> PENDING: 创建任务
    PENDING --> RUNNING: asyncio.Task 启动
    RUNNING --> COMPLETED: 执行成功
    RUNNING --> FAILED: 执行异常
    RUNNING --> CANCELLED: 用户取消 / 关闭超时
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

任务状态同时维护在 MySQL（持久化）和 Redis（实时查询）中。

### 8.5 RabbitMQ 架构

与 Java/Go 端对齐，采用相同的交换机和队列设计（`app/mq/` 模块）：

```mermaid
flowchart LR
    subgraph Producer["生产者 (dehaze-python)"]
        TaskService["TaskService._dispatch_task"]
        Publisher["Publisher (aio-pika)<br/>自动重连 + 指数退避"]
    end

    subgraph RabbitMQ["RabbitMQ Broker"]
        Exchange["dehaze.tasks<br/>(direct exchange)"]
        Q1["task.export<br/>导出任务队列<br/>(durable, TTL=24h)"]
        DLX["Dead Letter Exchange<br/>死信交换机"]
    end

    subgraph Consumer["消费者 (dehaze-python)"]
        ExportWorker["handle_export_task<br/>(handlers.py)"]
    end

    TaskService --> Publisher
    Publisher --> Exchange
    Exchange -->|"task.task.export"| Q1
    Q1 --> ExportWorker
    Q1 -.->|"nack/超时"| DLX
```

### 8.6 RabbitMQ 配置

```python
# Settings (config.py)
RABBITMQ_ENABLED: bool = False              # 是否启用
RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
RABBITMQ_EXCHANGE: str = "dehaze.tasks"      # 交换机（与 Go/Java 端一致）
RABBITMQ_EXCHANGE_TYPE: str = "direct"
RABBITMQ_ROUTING_KEY_PREFIX: str = "task"    # 路由键前缀
RABBITMQ_RECONNECT_MAX_RETRIES: int = 0     # 0 = 无限重试
RABBITMQ_RECONNECT_INITIAL_INTERVAL: float = 1.0
RABBITMQ_RECONNECT_MAX_INTERVAL: float = 30.0
RABBITMQ_PREFETCH_COUNT: int = 10           # 消费者预取数量
```

### 8.7 MQ 模块结构

```text
app/mq/
├── __init__.py        # 模块入口（导出 Publisher / Consumer / init_mq / close_mq）
├── publisher.py       # Publisher：消息发布、自动重连、指数退避
├── consumer.py        # Consumer：多队列消费、handler 注册、ack/nack
├── connection.py      # 全局单例管理、Lifespan 集成、handler 注册
└── handlers.py        # 消费者 handler（handle_export_task）
```

**已完成迁移**：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | 引入 `aio-pika`，实现 Publisher / Consumer（自动重连 + 指数退避） | ✅ 已完成 |
| **Phase 2** | 导出任务从 asyncio.Task 迁移为 MQ 优先 + fallback 双通道 | ✅ 已完成 |
| **Phase 3** | 与 Java/Go 端共享队列，实现跨语言任务协作 | 📋 规划中 |

### 8.8 Kafka 规划（日志管道）

与 Java/Go 端保持一致：

```mermaid
flowchart LR
    subgraph Sources["日志源"]
        AppLog["应用日志"]
        AuditLog["审计日志<br/>(操作日志)"]
        InferLog["推理日志"]
    end

    subgraph Kafka["Kafka Cluster"]
        Topic["dehaze.logs"]
    end

    subgraph Sink["日志存储"]
        ES["Elasticsearch<br/>(检索)"]
        S3["对象存储<br/>(归档)"]
    end

    Sources --> Topic --> Sink
```

**Python 端 Kafka 客户端选型**：`aiokafka`（原生 asyncio 支持，与 FastAPI 异步模型一致）。

---

## 9. 定时任务

### 9.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| **调度平台** | XXL-Job Admin | 与 Java/Go 端共享调度中心 |
| **Python 执行器** | pyxxl ≥ 0.4 | Python XXL-Job 执行器，原生 asyncio 支持 |
| **当前状态** | ✅ 已实现 | 执行器已集成，3 个定时任务已注册 |

### 9.2 架构设计

与 Java/Go 端共享 XXL-Job Admin，Python 端作为独立 Executor 注册：

```mermaid
flowchart LR
    subgraph XXLJob["XXL-Job Admin"]
        Scheduler["调度中心<br/>任务编排 / CRON / 日志 / 告警"]
    end

    subgraph JavaExecutor["dehaze-java Executor<br/>(port: 9999)"]
        J1["TaskCleanupHandler"]
        J2["XxlJobSampleHandler"]
    end

    subgraph GoExecutor["dehaze-go Executor"]
        G1["CleanupHandler"]
        G2["StatisticsHandler"]
    end

    subgraph PythonExecutor["dehaze-python Executor<br/>(port: 9998)"]
        P1["cleanupExpiredTasks<br/>过期任务清理"]
        P2["cleanupStuckTasks<br/>僵死任务回收"]
        P3["modelHealthCheck<br/>模型健康检查"]
    end

    Scheduler -->|"HTTP 回调触发"| JavaExecutor
    Scheduler -->|"HTTP 回调触发"| GoExecutor
    Scheduler -->|"HTTP 回调触发"| PythonExecutor
    JavaExecutor -->|"执行结果上报"| Scheduler
    GoExecutor -->|"执行结果上报"| Scheduler
    PythonExecutor -->|"执行结果上报"| Scheduler
```

### 9.3 已注册任务清单

| 任务名 | CRON 建议 | 功能 | 状态 |
|--------|-----------|------|------|
| `cleanupExpiredTasks` | `0 0 2 * * ?` | 删除 7 天前已完成/取消任务、30 天前所有任务，清理 Redis 缓存 | ✅ 已实现 |
| `cleanupStuckTasks` | `0 0 * * * ?` | 将超过 24h 的 pending/processing 任务标记为 failed | ✅ 已实现 |
| `modelHealthCheck` | `0 */30 * * * ?` | 检查 GPU 可用性/显存使用率、DB 连接、Redis 连接 | ✅ 已实现 |

### 9.4 XXL-Job 配置

```python
# Settings (config.py)
XXLJOB_ENABLED: bool = False                              # 是否启用
XXLJOB_ADMIN_URL: str = "http://localhost:8080/xxl-job-admin/api/"
XXLJOB_ACCESS_TOKEN: str = "default_token"
XXLJOB_EXECUTOR_APP_NAME: str = "xxl-job-executor-dehaze-python"
XXLJOB_EXECUTOR_HOST: str = "0.0.0.0"
XXLJOB_EXECUTOR_PORT: int = 9998                          # 区别于 Java 端 9999
```

### 9.5 Job 模块结构

```text
app/job/
├── __init__.py        # 模块入口（导出 init_xxljob / close_xxljob）
├── executor.py        # 执行器生命周期管理（PyxxlRunner daemon 模式）
└── handlers.py        # 定时任务 handler（cleanupExpiredTasks / cleanupStuckTasks / modelHealthCheck）
```

### 9.6 迁移进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | 部署 XXL-Job Admin（与 Java/Go 端共享） | ✅ 已完成 |
| **Phase 2** | 引入 pyxxl 执行器，注册到调度中心 | ✅ 已完成 |
| **Phase 3** | 实现任务清理、僵死回收、模型健康检查 | ✅ 已完成 |
| **Phase 4** | 新增缓存预热、统计报表等高级定时任务 | 📋 规划中 |

---

## 10. 日志系统

### 10.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 日志框架 | Python `logging` | 标准库，生态兼容性好 |
| 日志轮转 | RotatingFileHandler / TimedRotatingFileHandler | 按大小（10MB）或按天切割 |
| 编码处理 | 自研 `UTF8RotatingFileHandler` | 确保中文日志正确输出 |

### 10.2 日志配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 格式 | `%(asctime)s - %(levelname)s --- [%(thread)d] %(name)s : %(message)s` | 含时间、级别、线程 ID、模块名 |
| 文件路径 | `logs/app.log` | 日志目录 |
| 按大小切割 | 10MB / 保留 5 份 | `UTF8RotatingFileHandler` |
| 按天切割 | 每天午夜 | `UTF8TimedRotatingFileHandler` |
| 控制台输出 | 同时输出 | 开发环境调试用 |

### 10.3 操作日志（结构化审计）

通过 `OperationLogMiddleware`（Starlette BaseHTTPMiddleware）实现全链路操作日志：

| 特性 | 说明 |
|------|------|
| 写入方式 | `asyncio.create_task()` 异步写入 MySQL |
| 敏感字段过滤 | password / token / secret / authorization 等自动脱敏 |
| 排除路径 | `/health`、`/docs`、`/redoc`、`/openapi.json`、`/metrics`、`/favicon.ico` |
| 请求体截断 | 最大 500 字符 |
| 记录内容 | Method、Path、Status、Latency(ms)、IP、UserAgent、请求体、响应体 |

### 10.4 日志演进规划

```
当前: Python logging → 文件（按大小/天切割）
规划: logging → Kafka → Elasticsearch（检索）/ 对象存储（归档）
```

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | 结构化日志 | 切换到 JSON 格式，便于日志平台采集 |
| P2 | Kafka 日志管道 | 通过 aiokafka 将日志异步发送到 Kafka |
| P3 | 操作日志规范化 | 统一操作日志字段，与 Java/Go 端对齐 |

---

## 11. HTTP 服务与中间件

### 11.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI ≥ 0.115 | ASGI 异步框架，原生 async/await |
| ASGI 服务器 | uvicorn (标准模式) | 高性能 ASGI 服务器 |
| API 文档 | FastAPI 内置 (OpenAPI 3.1) | `/docs` (Swagger UI) + `/redoc` (ReDoc) |
| WebSocket | FastAPI 原生 WebSocket | 无需第三方库 |
| Schema 校验 | Pydantic 2.x | 请求/响应自动校验和文档生成 |

### 11.2 中间件链

请求经过的处理层按注册顺序（逆序执行）：

```mermaid
flowchart LR
    Req["请求"] --> CORS["CORS 跨域<br/>CORSMiddleware"]
    CORS --> OpLog["操作日志<br/>OperationLogMiddleware"]
    OpLog --> Prometheus["Prometheus 指标<br/>(条件启用)"]
    Prometheus --> JWT["JWT 认证<br/>Depends(get_current_user)"]
    JWT --> Permission["权限校验<br/>PermissionChecker"]
    Permission --> Handler["业务处理"]
```

### 11.3 中间件清单

| 组件 | 类型 | 功能 | 作用范围 |
|------|------|------|----------|
| CORSMiddleware | Starlette 中间件 | 跨域资源共享，开发/生产环境配置不同 Origin | 全局 |
| OperationLogMiddleware | BaseHTTPMiddleware | 请求/响应全链路记录（异步写入） | 全局（排除健康检查等路径） |
| HTTP Metrics | starlette-exporter | Prometheus HTTP 指标采集 | 全局（条件启用） |
| `get_current_user` | FastAPI Depends | JWT Token 验证、UserContext 注入、Token 黑名单检查 | 受保护路由 |
| `PermissionChecker` | FastAPI Depends 工厂 | RBAC 权限校验（支持通配符匹配） | 受保护路由 |
| `require_permission` | 函数装饰器 | 权限检查（装饰器风格，替代方案） | 受保护路由 |

### 11.4 CORS 配置

| 环境 | 允许 Origin |
|------|-------------|
| 开发 | `localhost:3000/5173/8080`、`127.0.0.1:3000/5173/8080` |
| 生产 | `localhost:3000/8080` |

### 11.5 路由注册

采用 FastAPI APIRouter 模式，在 `app/route/__init__.py` 集中注册：

| 路由模块 | 路径前缀 | 说明 |
|----------|----------|------|
| auth | `/api/v1/auth` | 认证（登录/注销/验证码） |
| user | `/api/v1/users` | 用户管理 |
| role | `/api/v1/roles` | 角色管理 |
| menu | `/api/v1/menus` | 菜单管理 |
| dept | `/api/v1/depts` | 部门管理 |
| dict | `/api/v1/dicts` | 字典管理 |
| dataset | `/api/v1/datasets` | 数据集管理 |
| algorithm | `/api/v1/algorithms` | 算法管理 |
| file | `/api/v1/files` | 文件管理 |
| task | `/api/v1/tasks` | 导出任务 |
| websocket | `/ws` | WebSocket 实时通信 |

---

## 12. 安全基础设施

### 12.1 认证体系

| 组件 | 实现 | 说明 |
|------|------|------|
| JWT | python-jose (HS256) | AccessToken + RefreshToken |
| Token 验证 | `Depends(get_current_user)` | 自动解析 → UserContext |
| Token 黑名单 | Redis `token:blacklist:{jti}` | 注销时将 JTI 加入黑名单 |
| 密码加密 | bcrypt | 密码哈希 |
| 验证码 | Redis 存储 + Pillow 生成 | 可配置长度/尺寸/字体/干扰线/过期时间 |

**UserContext 结构**：

```python
class UserContext(BaseModel):
    id: int
    username: str
    nickname: Optional[str]
    roles: list[str]
    permissions: list[str]

    @property
    def is_root(self) -> bool:
        return "ROOT" in self.roles
```

### 12.2 权限体系

提供两种权限校验方式（依赖注入 + 装饰器），适应不同场景：

**方式 1：依赖注入（推荐）**

```python
@router.post("/users")
async def create_user(
    _: None = PermissionChecker("sys:user:add"),
):
    ...
```

**方式 2：装饰器**

```python
@router.post("/users")
@require_permission("sys:user:add")
async def create_user(user: UserContext, ...):
    ...
```

权限匹配支持通配符：

```text
用户 → 角色（多对多） → 权限标识（多对多）
权限格式: 模块:功能:操作（如 sys:user:add）
通配符: * 匹配所有（ROOT 用户自动跳过校验）
fnmatch 双向匹配: sys:user:* ↔ sys:user:add
```

### 12.3 安全防护

| 防护类型 | 实现方式 | 说明 |
|----------|----------|------|
| SQL 注入 | SQLAlchemy 参数化查询 | ORM 层面天然防护 |
| XSS | Pydantic 输入校验 | Schema 层面类型约束 |
| CSRF | JWT Token 认证（非 Cookie） | API 接口无需 CSRF 保护 |
| CORS | CORSMiddleware | 限制允许的 Origin |
| 暴力破解 | 验证码 + 密码错误计数（规划） | 登录验证码 |

---

## 13. 统一响应与错误处理

### 13.1 响应格式

```json
{
  "code": "00000",
  "msg": "success",
  "data": { ... }
}
```

通过 `utils/result.py` 提供泛型 `Result[T]` 和工厂函数：`success()` / `error()` / `warning()`。

### 13.2 错误码体系

与 Java/Go 端共用同一套错误码规范（详见 [API 规范](./04-API规范.md)）：

| 前缀 | 类别 | 示例 |
|------|------|------|
| `00` | 成功 | `00000` - 一切 ok |
| `A0` | 用户端错误 | `A0001` - 用户端错误, `A0200` - 登录异常, `A0400` - 参数错误 |
| `B0` | 系统执行错误 | `B0001` - 系统执行出错, `B0210` - 并发限流 |
| `C0` | 第三方服务错误 | `C0001` - 调用第三方服务出错, `C0300` - 数据库服务出错 |

### 13.3 全局异常处理

通过 `register_exception_handlers(app)` 注册 FastAPI 全局异常处理器：

```mermaid
flowchart TB
    E1["BusinessException"] -->|400| R1["自定义 ResultCode"]
    E2["RequestValidationError"] -->|400| R2["PARAM_ERROR<br/>提取首个校验错误"]
    E3["JWTError"] -->|401| R3["TOKEN_INVALID"]
    E4["SQLAlchemyError"] -->|500| R4["DATABASE_ERROR<br/>脱敏后返回"]
    E5["Exception (兜底)"] -->|500| R5["SYSTEM_EXECUTION_ERROR<br/>开发环境含类型名"]
```

---

## 14. 实时通信

### 14.1 WebSocket 方案

基于 FastAPI 原生 WebSocket 实现，不依赖第三方库：

| 端点 | 协议 | 认证方式 |
|------|------|----------|
| `/ws?token=JWT_TOKEN` | 原生 WebSocket | URL Query 参数传递 JWT |

### 14.2 消息类型

| 事件 | 方向 | 用途 |
|------|------|------|
| `connected` | Server → Client | 连接成功确认 |
| `ping` / `pong` | 双向 | 心跳检测 |
| `broadcast` | Server → Client | 广播消息（推理进度、任务状态） |
| `private_message` | Server → Client | 私信消息 |
| `user_online` / `user_offline` | Server → Client | 用户上下线通知 |
| `shutdown_notification` | Server → Client | 服务关闭通知 |

---

## 15. 可观测性

### 15.1 当前状态

| 能力 | 状态 | 说明 |
|------|------|------|
| 应用日志 | ✅ 已实现 | 文件 + 控制台双输出 |
| 操作审计日志 | ✅ 已实现 | 全链路请求记录，异步写入 MySQL |
| 健康检查 | ✅ 已实现 | `GET /health`、`/health/db`、`/health/redis` |
| Prometheus 指标 | ✅ 已实现 | HTTP / GPU / 推理 / 任务 四大类指标 |

### 15.2 健康检查端点

| 端点 | 功能 | 返回信息 |
|------|------|----------|
| `GET /health` | 应用存活探针 | status / app / version |
| `GET /health/db` | 数据库连接检查 | `SELECT 1` 验证连接 |
| `GET /health/redis` | Redis 连接检查 | 连接状态 + 延迟(ms) |

### 15.3 Prometheus 指标体系

通过 `app/metrics/` 模块提供四大类指标：

| 指标类别 | 模块 | 关键指标 |
|----------|------|----------|
| **HTTP 指标** | `http_metrics.py` | 请求量、请求延迟、状态码分布 |
| **GPU 指标** | `gpu_metrics.py` | GPU 利用率、显存使用、温度、功耗 |
| **推理指标** | `inference_metrics.py` | 推理耗时（Histogram）、推理请求计数 |
| **任务指标** | `task_metrics.py` | 任务队列深度、处理中任务数、处理耗时 |

GPU 指标采集器作为后台任务运行，通过配置控制采集间隔：

```python
PROMETHEUS_ENABLED = True
PROMETHEUS_GPU_COLLECT_INTERVAL = 5  # 秒
```

### 15.4 可观测性演进规划

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | Prometheus endpoint (`/metrics`) | 暴露指标端点供 Prometheus 抓取 |
| P2 | Grafana Dashboard | 预置 GPU 利用率、推理吞吐、任务积压等面板 |
| P3 | 分布式追踪 | OpenTelemetry 集成（与 Java/Go 端统一 Trace ID） |

---

## 16. 技术栈总览

| 分类 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Python | ≥ 3.10 | 后端开发 + AI 推理 |
| **Web 框架** | FastAPI | ≥ 0.115 | ASGI 异步 HTTP 路由、中间件、WebSocket |
| **ASGI 服务器** | uvicorn | ≥ 0.32 | 高性能异步服务器 |
| **API 文档** | FastAPI 内置 OpenAPI 3.1 | - | Swagger UI + ReDoc 自动生成 |
| **ORM** | SQLAlchemy | ≥ 2.0 | 异步 ORM，声明式模型 |
| **数据库驱动** | aiomysql + PyMySQL | - | 异步/同步 MySQL 驱动 |
| **数据库迁移** | Alembic | - | Schema 版本管理 |
| **缓存** | redis-py (asyncio) | ≥ 6.4 | 异步 Redis 客户端 |
| **对象存储** | MinIO Python SDK | ≥ 7.2 | 文件/图像存储 |
| **认证** | python-jose | ≥ 3.3 | JWT Token (HS256) |
| **密码加密** | bcrypt | ≥ 4.0 | 密码哈希 |
| **Schema 校验** | Pydantic | ≥ 2.0 | 请求/响应校验、配置管理 |
| **配置管理** | pydantic-settings | ≥ 2.0 | 环境变量绑定、多环境配置 |
| **监控** | prometheus-client + starlette-exporter | - | Prometheus 指标采集 |
| **AI 推理** | PyTorch + torchvision | ≥ 2.9 | 深度学习推理引擎 |
| **HTTP 客户端** | httpx | ≥ 0.28 | 异步 HTTP 客户端 |
| **容器** | Docker (NVIDIA CUDA 12.1) | - | GPU 推理容器化 |
| **包管理** | uv | - | 快速 Python 包管理器 |
| **测试** | pytest + pytest-asyncio | ≥ 8.0 | 异步单元/集成测试 |
| **消息队列** | aio-pika (RabbitMQ) | ≥ 9.4 | 异步任务分发（MQ 优先 + asyncio.Task fallback） |
| **日志管道** | Kafka（规划） | - | 日志收集与流处理 |
| **定时任务** | pyxxl (XXL-Job) | ≥ 0.4 | 分布式定时任务调度（与 Java/Go 共享 Admin） |

---

## 17. Python 与 Java/Go 基础设施对照

| 基础设施能力 | dehaze-python | dehaze-java | dehaze-go | 一致性 |
|-------------|---------------|-------------|-----------|--------|
| **HTTP 框架** | FastAPI (uvicorn) | Spring MVC (Tomcat) | Gin | 接口语义一致 |
| **ORM** | SQLAlchemy 2.0 (异步) | MyBatis-Plus | GORM | 功能对等 |
| **Repository** | BaseRepository 泛型基类 | Mapper 层 | Repository 接口 | 功能对等 |
| **缓存** | Redis (单级 + 熔断器) | Spring Cache + Redis | 多级缓存 (FreeCache + Redis) | Python 端规划引入多级 |
| **消息队列** | ✅ aio-pika RabbitMQ (MQ优先+fallback) | @Async → RabbitMQ(规划) | ✅ 已实现 RabbitMQ | Python/Go 已对齐 |
| **定时任务** | ✅ pyxxl XXL-Job (3个任务已注册) | @Scheduled + XXL-Job | Ticker → XXL-Job | 三端共享 XXL-Job Admin |
| **日志** | Python logging | SLF4J + Logback | Zap | 格式/级别统一 |
| **日志管道** | 未实现 → Kafka(规划) | 未实现 → Kafka(规划) | 未实现 → Kafka(规划) | 统一规划 |
| **认证** | python-jose JWT | Spring Security + JWT | 自研中间件 + JWT | Token 格式互通 |
| **权限** | RBAC (Depends + 装饰器) | RBAC (@PreAuthorize) | RBAC (中间件) | 权限标识一致 |
| **数据权限** | SQLAlchemy 查询装饰(规划) | MyBatis-Plus 拦截器 | GORM Scopes | 语义一致 |
| **自动填充** | SQLAlchemy Event + ContextVar | MetaObjectHandler | GORM Callback | 字段名一致 |
| **API 文档** | FastAPI OpenAPI 3.1 | Knife4j (OpenAPI 3) | Swagger (swag) | 规范一致 |
| **监控** | prometheus-client | Micrometer + Prometheus | client_golang | 指标命名统一 |
| **健康检查** | `/health` + `/health/db` + `/health/redis` | Actuator `/actuator/health` | `/health` | 功能对等 |
| **错误码** | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | ✅ 完全一致 |
| **响应格式** | `{code, msg, data}` | `{code, msg, data}` | `{code, msg, data}` | ✅ 完全一致 |
| **WebSocket** | FastAPI 原生 WebSocket | STOMP + SockJS | — | Python/Java 各有实现 |
| **对象存储** | MinIO 直连 | MinIO / 阿里云 OSS | 通过 API | Python 端需直接处理图像 |
| **Redis 弹性** | 降级 + 重试 + 熔断器 | Lettuce 自动重连 | 自研重试 | Python 端更完善 |
| **并发模型** | asyncio 协程 + 多 Worker | 线程池 + Tomcat 线程 | Goroutine | 各语言最优实践 |
