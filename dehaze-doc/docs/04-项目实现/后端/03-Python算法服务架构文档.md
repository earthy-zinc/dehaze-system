# Python 后端架构 (dehaze-python)

图像去雾系统的 Python 后端，基于 FastAPI + Uvicorn 构建，承担**双重角色**：一是与 Java/Go 对等的业务后端（提供用户/权限/订单/会员/数据集/算法管理等完整业务 API），二是算法服务（基于 PyTorch 承载深度学习推理与图像质量评估）。两种职责同进程部署，共用同一套分层架构与基础设施。

> 构建/运行/测试说明见项目根目录的 `README.md`。

## 一、分层架构

```mermaid
flowchart TB
    subgraph External["外部请求"]
        Client["HTTP Client"]
        WS["WebSocket Client"]
    end

    subgraph Middleware["中间件 / 依赖注入链"]
        direction LR
        CORS["CORS 跨域"]
        Prometheus["Prometheus 指标"]
        IPBlacklist["IP 黑名单"]
        OpLog["操作日志"]
        Trace["TraceID"]
        SessionAuth["Session 认证 (get_current_user)"]
        Permission["权限校验 (@require_permission)"]
    end

    subgraph Router["Router 层 (app/router/)"]
        direction LR
        R1["参数绑定 (Pydantic)"]
        R2["参数校验"]
        R3["响应封装"]
    end

    subgraph Service["Service 层 (app/service/)"]
        direction LR
        S1["业务编排"]
        S2["缓存交互"]
        S3["存储/任务策略"]
        S4["推理服务 (PyTorch)"]
    end

    subgraph Repository["Repository 层 (app/repository/)"]
        direction LR
        Repo1["BaseRepository 泛型 CRUD"]
        Repo2["子类扩展特定查询"]
    end

    subgraph Infra["基础设施层"]
        direction LR
        MySQL[("MySQL")]
        Redis[("Redis")]
        MinIO[("MinIO")]
        RabbitMQ[("RabbitMQ")]
    end

    Client --> Middleware --> Router --> Service --> Repository --> MySQL
    Service --> Infra
    WS --> Service
```

### 层级职责

| 层级 | 包路径 | 职责 |
|------|--------|------|
| 中间件/依赖注入 | `middleware/` + `dependencies/` + `decorators/` | 请求拦截、认证、鉴权、限流、防重提交、操作日志、TraceID、IP 黑名单 |
| Router 层 | `router/` | 参数绑定与校验（Pydantic）、调用 Service、统一响应封装 |
| Service 层 | `service/` | 业务逻辑编排、缓存交互、存储/任务策略选择、异步任务分发 |
| Repository 层 | `repository/` | 数据库 CRUD 封装、分页、模糊搜索、批量操作、复杂查询 |
| Models 层 | `models/` | ORM 实体、Schema 定义、Enum 常量 |
| Core 层 | `core/` | 统一错误码、响应封装、业务异常 |
| 基础设施层 | `infrastructure/` | 第三方客户端连接单例与生命周期（MySQL/Redis/Mongo/MinIO/ES/MQ）、日志、缓存、定时任务、指标采集、**AI 技术资源客户端（LLM/Embedding/语音引擎）**；不感知 FastAPI 与业务 |

### 基础设施分层规范（连接 / 注入 / 业务）

基础设施按"连接管理、依赖注入、业务服务"三层收敛，统一管理边界：

| 类别 | 位置 | 职责 | 示例 |
|------|------|------|------|
| 连接管理 | `infrastructure/` | 第三方客户端单例与生命周期，不感知 FastAPI | `infrastructure/storage/minio_client.py`（`get_minio_client`/`minio_executor`）、`infrastructure/es/`（`es_client` 单例）、`infrastructure/mq/` |
| 依赖注入 | `dependencies/` | FastAPI `Depends` 提供器，包装连接供 Router/Service 使用 | `dependencies/auth.py`、`dependencies/redis.py`、`dependencies/mongo.py` |
| 业务服务 | `service/` | 业务编排，通过注入或基础设施单例使用客户端，禁止自行创建连接 | `service/file_service.py`、`service/storage/`（存储策略实现） |

规范约束：

- 单向依赖：`service → dependencies → infrastructure`；service 可直接使用基础设施单例，但禁止在 service 中直接 `Minio(...)` 等新建第三方连接
- 客户端实例全局唯一，统一封装在 `infrastructure/` 下，供各层复用
- ES 仅在 service 层内部使用、无请求上下文依赖，采用"infrastructure 封装 + 模块级单例"模式，不经过依赖注入
- Redis/Mongo 采用"连接 + 注入一体化"（`dependencies/redis.py`、`dependencies/mongo.py`），属 FastAPI 标准实践，保持现状
- `database.py` 的引擎/会话工厂/Base 属应用级基础设施，与 `config.py` 同类放根目录；`get_db` 为依赖注入提供器
- **AI 技术资源**：LLM/Embedding/语音引擎等外部模型能力统一收敛到 `infrastructure/` 子目录（`llm/`、`embedding/`、`voice/`），service 层只做业务编排与路由决策，不直接持有协议转换/子进程管理/Key 轮换实现（详见 3.11 AI 模型基础设施）

### 依赖注入策略

基于 FastAPI 原生依赖注入系统：

- 使用 `Depends()` 声明依赖关系，由框架自动解析
- 数据库 Session 通过 `get_db` 异步生成器注入，自动管理生命周期
- Redis 连接通过 `get_redis` 生成器注入
- 权限校验通过 `@require_permission` 装饰器实现

```mermaid
flowchart LR
    Router["Router Handler"] --> Depends["Depends(get_current_user)"]
    Depends --> SessionAuth["Session 校验 -> UserContext"]
    Router --> Depends2["Depends(get_db)"]
    Depends2 --> Session["AsyncSession"]
    Router --> Depends3["@require_permission('sys:user:add')"]
    Depends3 --> Check["权限校验"]
```

### 数据模型分层

| 模型类型 | 包路径 | 职责 | 示例 |
|----------|--------|------|------|
| Entity | `models/entity/` | 数据库表映射，SQLAlchemy Column 定义 | `SysUser` |
| Schema | `models/schema/` | API 请求/响应 Schema，Pydantic 校验 + OpenAPI 自动生成 | `UserPageQuery` |
| Enum | `models/enum/` | 枚举常量定义 | `TaskStatus` |

## 二、项目目录结构

```
dehaze-python/
├── app/                               # 主应用目录
│   ├── __init__.py                    # 延迟导入入口
│   ├── main.py                        # FastAPI 应用入口
│   ├── lifecycle.py                   # Lifespan 上下文管理器
│   ├── config.py                      # 多环境配置（Pydantic Settings）
│   ├── database.py                    # SQLAlchemy 2.0 异步引擎 & Session 工厂
│   ├── core/                          # 核心通用层（错误码/响应封装/异常）
│   ├── dependencies/                  # FastAPI 依赖注入（auth/redis）
│   ├── decorators/                    # 横切关注点装饰器（permission/rate_limit/repeat_submit）
│   ├── middleware/                    # ASGI 中间件（trace/operation_log/ip_blacklist/non_null_response）
│   ├── infrastructure/               # 基础设施层（连接单例 + AI 技术资源客户端）
│   │   ├── llm/                      #   LLM：model_client(协议工厂)/openai_compat_client/anthropic_client/model_registry(路由)/provider_key_selector(Key 轮换)/model_seeder(播种)/local_llm_manager(本地子进程)
│   │   ├── embedding/                #   Embedding：embedding_client（端点由 sys_ai_provider.api_base_url 配置化派生）
│   │   ├── voice/                    #   语音引擎：funasr_client/funasr_engine(ASR)、piper_tts_engine(TTS，进程内推理)
│   │   └── ...                       #   storage/es/mq/redis/mongo 等外部连接单例
│   ├── models/                        # 数据模型（entity/schema/enum）
│   ├── repository/                    # Repository 层（BaseRepository 泛型基类）
│   ├── router/                        # 路由层（30+ 个业务 APIRouter + health/metrics + WebSocket）
│   ├── service/                       # 服务层（prediction/task_tracker/file/storage/import_export/member/order 等）
│   └── utils/                         # 工具层（password/file/datetime/tree 等）
├── algorithm/                         # 去雾算法模块（30 种算法，平铺结构）
├── config.py                          # 算法模块配置
├── migrations/                        # Alembic 数据库迁移
├── tests/                             # 测试
├── pyproject.toml                     # 项目依赖（uv 管理）
├── Dockerfile                         # GPU 推理容器化
└── logs/                              # 运行时日志目录
```

## 三、核心模块

### 3.1 业务模块概览

Python 端实现与 Java/Go 对等的完整业务层，按领域分组如下（各模块的需求规格、API 契约与实现详见 [03-模块设计](../../03-模块设计/)）：

| 领域 | 覆盖模块 |
|------|---------|
| 认证授权 | 登录/登出、Session、API Key、验证码 |
| 系统管理 | 用户、角色、部门、菜单、字典 |
| 去雾业务 | 图像输入、算法选择、预测处理、效果对比、评估指标、任务、收藏、预设、推荐 |
| 数据与算法管理 | 数据集、数据项、算法管理、导入导出 |
| 商业化 | 会员、订单、套餐、优惠券、支付、退款、通知设置 |
| 消息通知 | 站内消息、消息模板、公告、反馈、WebSocket 推送 |
| 文件与运维 | 文件管理、健康检查、监控指标、客户端日志 |

### 3.2 算法模块

`algorithm/` 目录平铺 30 种去雾算法（如 RIDCP、WPXNet、Dehamer 等），每个算法独立目录，统一以 `run.py` 为入口、由 `model_loader.py` 加载，通过 `importPath` 配置动态选择。各算法的模型结构与论文依据详见[算法/学术论文文档](../算法/学术论文文档.md)。

### 3.3 算法管线

```mermaid
flowchart LR
    Input["图像输入"] --> Validate["参数校验<br/>(Pydantic Schema)"]
    Validate --> Interceptor["拦截器链<br/>(WPXNet预查询等)"]
    Interceptor -->|命中| Return1["直接返回缓存结果"]
    Interceptor -->|未命中| Download["下载原始图像"]
    Download --> MD5["计算 MD5"]
    MD5 --> CacheCheck{"Redis 缓存<br/>命中?"}
    CacheCheck -->|命中| Return2["返回缓存结果"]
    CacheCheck -->|未命中| Inference["PyTorch 推理<br/>(run_in_executor)"]
    Inference --> Upload["上传结果到 MinIO"]
    Upload --> Cache["写入 Redis 缓存"]
    Cache --> Return3["返回结果"]
```

**推理线程池**：PyTorch 推理为 CPU/GPU 密集型同步操作，通过 `ThreadPoolExecutor` 在事件循环外执行以避免阻塞。并发数由 `INFERENCE_THREAD_POOL_SIZE` 配置项控制（默认 2），可按 GPU 显存/卡数调整（单卡 24GB 建议 1-2），无需改代码即可适配不同部署环境。

**后台任务追踪**：预测（`prediction_service`）、评估（`evaluation_service`）、对比（`compare_service`）三个核心推理服务在提交 `asyncio.create_task` 后台任务后，均注册到 `TaskTracker`（`task_id` 形如 `pred:{log_id}` / `eval:{log_id}` / `compare:{task_id}`），与导出/下载任务统一纳入优雅关闭与全局任务视图。注册失败不影响主流程（降级为日志告警），与 `task_service` 行为一致。`TaskTracker.initiate_shutdown()` 在 30s 超时窗口内等待这些推理任务完成，避免 Worker 崩溃或重启时结果文件半写入。

### 3.3 预测流程插件化（拦截器链）

预测主流程通过责任链模式支持可插拔拦截器：

| 组件 | 职责 |
|------|------|
| `PredictionContext` | 请求上下文（algorithm/file_id/image_url/origin_file/params） |
| `InterceptedResult` | 拦截命中后返回的结果 |
| `PredictionInterceptor` (ABC) | 拦截器抽象基类 |
| `PredictionInterceptorChain` | 责任链：按注册顺序执行，第一个命中即短路 |
| `WpxNetPredictionInterceptor` | WPXNet 预查询：通过 `sys_wpx_file` 表查表短路 |

WPXNet 预查询命中后直接返回结果，跳过 PyTorch 推理，响应时延从秒级降至毫秒级。

### 3.5 存储抽象层

策略模式适配多存储后端：

```mermaid
flowchart LR
    Factory["StorageFactory<br/>按 FILE_STORAGE_TYPE 切换"] --> MinIO["MinIO 存储"]
    Factory --> Local["本地文件存储"]
```

`sys_file` 表只存 `object_name` + `storage`，URL 运行时拼接不落库。

### 3.6 通用导入导出

Handler 模式 + 通用策略实现：

| 模块 | ExportHandler | ImportHandler |
|------|--------------|--------------|
| 用户管理 | UserExportHandler | UserImportHandler |
| 角色管理 | RoleExportHandler | RoleImportHandler |
| 部门管理 | DeptExportHandler | DeptImportHandler |
| 菜单管理 | MenuExportHandler | MenuImportHandler |
| 字典管理 | DictExportHandler | DictImportHandler |
| 数据集管理 | DatasetExportHandler | -（仅导出） |
| 算法管理 | AlgorithmExportHandler | AlgorithmImportHandler |

### 3.7 消息队列

采用 MQ 优先 + asyncio.Task fallback 双通道架构：

```mermaid
flowchart TB
    subgraph Dispatch["任务分发"]
        Create["创建任务 (写入 DB + Redis)"]
        Check{"RabbitMQ 可用?"}
    end

    subgraph MQPath["MQ 通道 (持久化)"]
        Publish["Publisher.publish()"]
        Queue["task.export 队列 (durable, TTL=24h)"]
        Handler["handle_export_task Consumer 回调"]
    end

    subgraph FallbackPath["Fallback 通道 (进程内)"]
        AsyncTask["asyncio.create_task()"]
        Tracker["TaskTracker 注册"]
    end

    Create --> Check
    Check -->|是| Publish --> Queue --> Handler
    Check -->|否| AsyncTask --> Tracker
```

与 Java/Go 端共享 `dehaze.tasks` (direct exchange) 交换机。

### 3.8 图像特征分析服务

为推荐管理模块提供图像特征提取能力，输出 7 维结构化特征向量供推荐引擎规则匹配：

| 特征维度 | 权重 | 实现方式 |
|---------|:----:|---------|
| 雾霾浓度 | 30% | 暗通道先验估计 + 透射率计算 |
| 场景类型 | 20% | ResNet 预训练模型场景分类（城市/风景/建筑/夜景/逆光/室内） |
| 光照条件 | 15% | 亮度直方图分析 + 曝光评估 |
| 图像复杂度 | 10% | 边缘密度（Canny）+ 纹理丰富度（LBP） |
| 颜色分布 | 10% | 色温估计 + 饱和度统计 + HSV 直方图 |
| 分辨率 | 5% | 尺寸归一化分类（标清/高清/超清） |
| 噪声水平 | 10% | 局部方差估计 + 频域分析 |

服务通过 HTTP 接口接收图像 URL，内部复用 PyTorch 推理管线（`run_in_executor` 避免阻塞事件循环），特征分析结果按图像 MD5 缓存 1 小时。Java 后端的 `RecommendationServiceImpl.analyze` 通过 `PythonAlgorithmClient.analyzeImage`（POST `/api/v1/recommendations/analyze`，复用算法服务 HTTP 客户端的重试/熔断/幂等机制）调用此服务获取真实特征；Go 后端的 `RecommendationService.Analyze` 通过 `pkg/algorithm.Client.AnalyzeImage` 调用同一接口。两端在 Python 服务不可用时均返回错误，不降级为伪特征。

**放在 Python 端而非 Java/Go 的理由**：图像特征分析依赖 PyTorch 预训练模型（场景分类）和 OpenCV 图像处理（暗通道、边缘检测、直方图），这些库为 Python 生态原生；复用已有推理管线和 GPU 资源，避免在 Java/Go 端重复引入图像处理依赖。

### 3.9 评估指标计算服务

为效果对比模块提供去雾效果量化评估能力，计算 5 项专业指标：

| 指标 | 说明 | 合格阈值 | 实现方式 |
|------|------|---------|---------|
| PSNR | 峰值信噪比 | ≥ 30.0 dB | 像素级 MSE -> 峰值比 |
| SSIM | 结构相似性 | ≥ 0.8 | 窗口滑动 + 亮度/对比度/结构三项 |
| LPIPS | 学习感知相似度 | ≤ 0.3 | 预训练 VGG/AlexNet 特征距离 |
| NIQE | 自然图像质量 | ≤ 5.0 | 统计模型 + MVN 距离 |
| Entropy | 信息熵 | 统计展示 | 灰度直方图 Shannon 熵 |

评估结果按 `{algorithmId}:{predMd5}:{refMd5}` 永久缓存，相同图片+相同算法的评估请求命中缓存直接返回。Java/Go 后端通过算法管理模块的 EvaluationService 委托调用此服务。

### 3.10 推荐生成完整管线

```mermaid
flowchart LR
    Input["图像输入"] --> FeatureAnalysis["图像特征分析<br/>7维特征向量"]
    FeatureAnalysis --> RuleMatch["规则匹配<br/>场景→算法映射"]
    RuleMatch --> Rank["综合排序<br/>特征匹配度+评分+成功率+采纳率"]
    Rank --> ColdStart["冷启动注入<br/>新算法随机曝光"]
    ColdStart --> Output["Top N 推荐结果<br/>含匹配度和推荐理由"]
```

推荐引擎的规则匹配和排序逻辑在 Java/Go 后端实现，Python 端仅负责图像特征向量提取。Python 端的特征分析服务作为推荐管线的第一环节，其输出直接决定推荐候选集。

### 3.11 AI 模型基础设施

LLM / Embedding / TTS / ASR 等模型能力统一作为**基础设施**管理，与业务编排解耦。核心原则：`infrastructure` 回答"如何与外部技术资源对话"（协议转换、子进程、Key 轮换），`service` 回答"业务上该用哪个模型、失败如何降级"（路由决策）。

```mermaid
flowchart TB
    subgraph Service["service/（业务编排，决策）"]
        LLM["llm_client（对话编排/降级）"]
        KB["知识库/记忆（embedding、rerank 调用）"]
        Voice["语音交互（ASR/TTS 调用）"]
    end
    subgraph Infra["infrastructure/（技术资源对话）"]
        MR["model_registry<br/>sys_ai_provider/sys_ai_model 路由"]
        KS["provider_key_selector<br/>Key 轮换/冷却/日额度(Redis)"]
        MC["model_client 工厂<br/>openai_compat/anthropic"]
        EC["embedding_client<br/>api_base_url 派生端点"]
        VE["voice/（FunASR、Piper 进程内引擎）"]
        MS["model_seeder<br/>本地模型幂等播种"]
    end
    subgraph Store["sys_ai_provider / sys_ai_model / sys_ai_provider_key"]
    end
    LLM --> MC
    LLM --> MR --> Store
    LLM --> KS --> Store
    KB --> EC --> Store
    KB --> MR --> Store
    Voice --> VE
    MS --> Store
```

**关键组件与职责**：

| 组件 | 位置 | 职责 |
|------|------|------|
| 统一模型客户端接口 | `infrastructure/llm/model_client.py` | `LlmStreamChunk`、鉴权头、`create_chat_client(protocol_type)` 按 `sys_ai_provider.protocol_type`（openai_compat / anthropic）工厂分发，屏蔽协议差异 |
| OpenAI 兼容客户端 | `infrastructure/llm/openai_compat_client.py` | SSE 流式、tool_call 聚合、reasoning_content 思考流 |
| Anthropic 客户端 | `infrastructure/llm/anthropic_client.py` | tool_use 三段式聚合、Prompt Caching、thinking 流 |
| 模型路由注册表 | `infrastructure/llm/model_registry.py` | `get_call_routes` 按 `sys_ai_model` 能力与状态解析降级链候选路由（能力不足时剔除并短路），路由决策完全由数据库配置驱动 |
| Key 选择器 | `infrastructure/llm/provider_key_selector.py` | 从 `sys_ai_provider_key` 按 优先级/权重/冷却/连续失败/日额度 选取 Key（Redis 状态），调用后回写成功/失败 |
| 本地模型播种器 | `infrastructure/llm/model_seeder.py` | 幂等播种 local provider / 占位 Key / 内置模型（qwen3-0.6b 对话 + qwen3-embedding-0.6b 向量登记），启动时由 `lifecycle.py` 调用 |
| Embedding 客户端 | `infrastructure/embedding/embedding_client.py` | 向量化/维度查询；端点由 provider 的 `api_base_url` **配置化派生**（`api_base_url + /embeddings`，cohere 特判 `/v1/embed`），新增 OpenAI 兼容供应商零代码 |
| 本地 LLM 子进程 | `infrastructure/llm/local_llm_manager.py` 等 | llama-cpp-python 子进程生命周期（拉起/健康检查/回收），对话与 Embedding 共用 `/v1` 端点 |
| 语音引擎 | `infrastructure/voice/` | FunASR（ASR）与 Piper（TTS）进程内推理，由 `config.py` 的 `VOICE_*` 配置驱动（引擎音色无供应商路由语义，不进入模型注册表） |

**配置化路由与播种边界**：

- 第三方供应商（qwen/openai/anthropic 等）：管理接口写入 `sys_ai_provider`/`sys_ai_model`/`sys_ai_provider_key`，运行时直接生效，**切换/升级/新增供应商无需改代码**
- 本地模型：`model_seeder.ensure_local_models` 启动时幂等播种 local provider 与占位 Key、内置 LLM（qwen3-0.6b，状态启用）与内置 Embedding（qwen3-embedding-0.6b，状态停用仅登记目录，避免进入对话模型列表）
- Embedding/Rerank 端点统一由 `sys_ai_provider.api_base_url` 派生（OpenAI 兼容 `/embeddings`、`/rerank`），与对话客户端共用同一供应商配置，不再维护独立的硬编码端点表

## 四、配置管理

| 组件 | 选型 | 说明 |
|------|------|------|
| 配置框架 | Pydantic Settings v2 | 类型安全、自动校验、环境变量绑定 |
| 环境变量 | `.env` 文件 + `os.getenv` | 敏感信息外部化 |
| 配置切换 | `APP_ENV` 环境变量 | 决定加载哪个配置类 |

多环境支持：

| 环境 | 配置类 | 特性差异 |
|------|--------|----------|
| 开发 | `DevelopmentSettings` | DEBUG=True，Session Cookie 关闭 Secure |
| 测试 | `TestingSettings` | DEBUG=True，独立测试数据库 `dehaze_test` |
| 生产 | `ProductionSettings` | 强制校验密码非空、CORS 禁止 localhost，JSON 日志 |

敏感信息通过 `DEHAZE_HOST` 和 `DEHAZE_PASSWORD` 统一管理，派生地址和连接串通过 `@property` 自动拼接。

## 五、安全认证

| 组件 | 实现 | 说明 |
|------|------|------|
| Session 认证 | Redis `session:{sessionId}` | TTL 7 天，剩余 < 24h 自动续期 |
| 用户上下文 | `Depends(get_current_user)` | 自动解析 Session -> UserContext |
| 密码加密 | bcrypt（`utils/password.py`） | 专用线程池异步执行 |
| 验证码 | Redis 存储 + Pillow 生成 | 可配置长度/尺寸/字体/干扰线 |

权限校验通过 `require_permission` 装饰器实现，支持通配符匹配（`sys:user:*` 匹配 `sys:user:add`），ROOT 用户自动跳过校验。

安全防护：

| 防护类型 | 实现方式 |
|----------|----------|
| SQL 注入 | SQLAlchemy 参数化查询 |
| XSS | `validate_no_xss` 输入校验（Pydantic Schema 层 + Service 层双重校验） |
| CSRF | Session Cookie（SameSite=Lax） |
| CORS | CORSMiddleware |
| 暴力破解 | 验证码 + IP 黑名单自动封禁 |
| 限流 | RateLimitMiddleware（Redis 固定窗口） |
| 防重复提交 | AntiRepeatMiddleware（ASGI 中间件，基于 user_id+method+uri+body_hash，Redis SET NX EX，默认 5 秒；排除文件上传/数据集/数据项等已有自身幂等机制的写接口） |

### 5.1 行级数据权限（DataScope）

与 Java（MyBatis-Plus `DataPermissionInterceptor`）、Go（GORM `dataScopeCallback`）对齐，Python 端基于 SQLAlchemy 2.0 异步 ORM 实现**显式过滤**方案：在需要数据权限的 Repository 查询中显式调用 `apply_data_scope(stmt, user, db, dept_field=..., creator_field=...)` 按当前用户 `data_scope` 追加 `WHERE` 条件。未采用 SQLAlchemy event 自动改写，因异步 Session 下 event 回调难以可靠获取当前请求的用户上下文（ContextVar 在 event 回调中不可靠）。

**data_scope 取值与过滤行为**（与 `sys_role.data_scope` 注释、Go `DataScope*` 常量一致）：

| 取值 | 含义 | 过滤条件 |
|:----:|------|---------|
| `NULL` / `0` | 全部数据 | 原样返回（ROOT 用户始终跳过过滤） |
| `1` | 部门及子部门 | `WHERE dept_field IN (本部门及子部门ID)`，子部门ID由 `dept_repository.get_children_ids` 查询 |
| `2` | 本部门 | `WHERE dept_field == user.dept_id`；用户无部门时返回空集 |
| `3` | 本人 | `WHERE creator_field == user.id` |

**调用约定**：

- Repository 查询方法新增 `current_user` 参数，由 Service 层从 `Depends(get_current_user)` 获取并透传
- `dept_field` 指向业务表的部门字段（如 `SysUser.dept_id`）；无部门字段的表（如订单、反馈）通过 JOIN `sys_user` 取 `dept_id` 实现"本部门"过滤，"本人"过滤使用 `creator_field`（如 `SysOrder.user_id`）
- 未知 `data_scope` 取值保守返回空集（`WHERE false()`）

**已接入的查询清单**：用户分页（`user_repository.get_page`）、订单分页（`order_repository.get_page`）。新增业务查询如涉及多租户可见性，须按同样方式接入 `apply_data_scope`。

## 六、数据访问层

| 组件 | 选型 | 说明 |
|------|------|------|
| ORM | SQLAlchemy 2.0 + 异步模式 | 声明式模型、AsyncSession |
| 数据库驱动 | aiomysql（异步）+ PyMySQL（同步） | 异步为主，同步用于 Alembic 迁移 |
| 数据库迁移 | Alembic (纯 CLI) | 版本化 Schema 管理 |
| 文档数据库 | Motor（MongoDB 异步） | 登录审计、业务操作审计（白名单驱动） |
| 对象存储 | MinIO Python SDK | 文件/图像存储 |

事务管理采用"请求边界 = 事务边界"模型，由 `get_db()` / `get_db_session()` 统一管理 commit/rollback。Router 层持有事务边界，Service 层做业务编排不 commit，Repository 层只做 flush。**约束**：每个 Router handler 对应一个工作单元，跨多 Repository 的原子提交须由同一 handler 内的单一 Session 包裹；Service 不得自行开新 Session 或跨请求持有 Session，以避免事务作用域泄漏。

Repository 层引入泛型 `BaseRepository[T]`，提供标准 CRUD 操作，子类只需声明 `model` 类型即可继承全部能力。

审计字段自动填充通过 SQLAlchemy event 事件机制 + ContextVar 实现：`before_insert` 填充 `create_time`/`update_time`/`create_by`/`update_by`，`before_update` 填充 `update_time`/`update_by`。只追加日志/流水/历史表（如 `sys_ai_credit_log`、`sys_ai_agent_thought`）继承 `AppendOnlyModel` 基类，仅自动填充 `create_time`，操作人语义由业务字段表达（`operator_id`/`auditor_id`）。

存储分工：

```mermaid
flowchart LR
    subgraph Relational["MySQL (结构化数据)"]
        User["用户/角色/部门/菜单/字典"]
        Business["数据集/算法/任务"]
        PredEvalLog["预测日志/评估日志"]
    end

    subgraph Document["MongoDB (审计日志)"]
        LoginLog["登录审计 (login_log)"]
        AuditLog["业务操作审计 (audit_log，白名单驱动)"]
    end

    subgraph Object["多存储后端"]
        Image["原始图像/去雾结果 (object_name + storage)"]
        Export["导出文件 (object_name + storage)"]
        Dataset["数据集静态文件 (nginx-static)"]
    end

    subgraph Cache["Redis (缓存与会话)"]
        Session["验证码/Session存储"]
        Task["任务状态/取消标志"]
        WS["WebSocket在线状态/Pub/Sub"]
    end
```

## 七、缓存体系

Redis 异步客户端作为缓存层，连接池管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_connections | 100 | 最大连接数 |
| socket_timeout | 5.0s | 操作超时 |
| health_check_interval | 30s | 健康检查间隔 |

Redis 弹性机制：

| 机制 | 说明 |
|------|------|
| 优雅降级 | `redis_operation_with_fallback()` Redis 不可用时执行 fallback |
| L1 本地缓存 | `local_cache.py`（TTLCache + SingleFlight）降低 Redis 压力 |

## 八、定时任务

通过 pyxxl (XXL-Job Python 执行器) 与 Java/Go 端共享调度中心，handler 命名与 Java/Go 端对齐（`autoRenew`/`resetMonthlyQuota` 等三端统一）：

| 任务名 | 功能 | 三端共有 |
|--------|------|:--------:|
| `cleanupExpiredTasks` | 删除过期任务，清理 Redis 缓存 | ✅ |
| `cleanupStuckTasks` | 将超过 30min 的 processing、24h 的 pending 异常任务标记为 failed | ✅ |
| `cleanupStuckPredEvalLogs` | 回收预测/评估僵尸任务（超 10min 未更新） | ✅ |
| `expireOrders` | 待支付订单超时自动取消，释放锁定优惠券 | ✅ |
| `completeExpiredOrders` | 已支付订单到期归档 | ✅ |
| `expireUserCoupons` | 用户优惠券过期失效 | ✅ |
| `retryFailedRefunds` | 退款失败记录重试（上限 3 次） | ✅ |
| `autoRenew` | 自动续费扣款（balance/wechat/alipay） | ✅ |
| `resetMonthlyQuota` | 每月 1 日重置会员月度配额 | ✅ |
| `processExpiredMembers` | 会员过期降级（按成长值重算等级） | ✅ |
| `sendExpireReminders` | 会员到期前 7/3/1 天推送续费提醒 | ✅ |
| `sendScheduledAnnouncements` | 发送定时公告 | ✅ |
| `cleanupExpiredMessages` | 清理过期消息（分批 500 条） | ✅ |
| `refreshUnreadCountCache` | 未读数缓存全量刷新 | ✅ |
| `modelHealthCheck` | 检查 GPU 可用性/显存使用率、DB/Redis 连接 | Python 专属 |
| `cleanupOrphanFiles` | 清理 MinIO 中无数据库记录关联的孤儿文件 | Python 专属 |
| `cleanupTempFiles` | 清理临时目录中过期的临时文件 | Python 专属 |

> 共 17 个 handler（14 个三端共有 + 3 个 Python 专属运维任务）。Java 独有的 `processDelayedPush`（DND 免打扰延迟推送）Python 端未实现，属独立架构差异（见 [Java 改造计划 §2.1](../../05-改造计划/Java后端架构改造计划.md)）。

```mermaid
flowchart LR
    subgraph XXLJob["XXL-Job Admin"]
        Scheduler["调度中心"]
    end

    subgraph PythonExecutor["dehaze-python Executor (port: 9998)"]
        P1["三端共有 Job x14"]
        P2["Python 专属 Job x3"]
    end

    Scheduler --> PythonExecutor
```

## 九、WebSocket 实时通信

基于 FastAPI 原生 WebSocket 实现，通过 Redis Pub/Sub 实现跨 Worker 通信：

```mermaid
flowchart TB
    subgraph Worker1["Worker 1"]
        WS1["WebSocket 连接"]
        Pub1["Redis Pub (dehaze:ws:broadcast)"]
    end
    subgraph Worker2["Worker 2"]
        WS2["WebSocket 连接"]
        Sub2["Redis Sub (dehaze:ws:broadcast)"]
    end
    subgraph Redis["Redis"]
        Channel["dehaze:ws:broadcast Pub/Sub 频道"]
        Online["dehaze:ws:online_users Sorted Set"]
    end
    WS1 --> Pub1 --> Channel
    Channel --> Sub2 --> WS2
    WS1 --> Online
    WS2 --> Online
```

## 十、应用生命周期

```mermaid
sequenceDiagram
    participant Uvicorn as uvicorn
    participant Lifespan as lifecycle.py
    participant DB as 数据库
    participant Redis as Redis
    participant Tracker as 任务追踪器
    participant MQ as RabbitMQ
    participant XXL as XXL-Job
    participant Seeder as model_seeder

    Uvicorn->>Lifespan: 启动 ASGI 应用
    Lifespan->>DB: init_db() 连接测试
    Lifespan->>Redis: check_redis_health()
    Lifespan->>Seeder: ensure_local_models()（幂等播种 local provider/Key/内置 LLM+Embedding）
    Seeder->>DB: sys_ai_provider / sys_ai_model / sys_ai_provider_key 补齐
    Lifespan->>Tracker: init_task_tracker() + start()
    Lifespan->>MQ: init_mq() (条件启用)
    Lifespan->>XXL: init_xxljob() (条件启用)
    Lifespan-->>Uvicorn: yield (应用就绪)
```

优雅关闭：TaskTracker 拒绝新任务 -> WebSocket 通知客户端 -> 等待运行中任务完成（30s 超时）-> 关闭 XXL-Job/RabbitMQ -> 关闭 Redis/数据库连接池。

本地开发统一通过项目根目录 `scripts/run.py` 管理三端后端的生命周期。

## 十一、统一响应与错误处理

错误码采用 5 位字符串编码，与 Java/Go 端保持一致：

| 前缀 | 类别 | 示例 |
|------|------|------|
| `00` | 成功 | `00000` - 一切 ok |
| `A0` | 用户端错误 | `A0001` - 用户端错误, `A0200` - 登录异常, `A0400` - 参数错误 |
| `B0` | 系统执行错误 | `B0001` - 系统执行出错, `B0210` - 并发限流 |
| `C0` | 第三方服务错误 | `C0001` - 调用第三方服务出错, `C0300` - 数据库服务出错 |

通过 `register_exception_handlers(app)` 注册全局异常处理器，统一拦截 BusinessException、RequestValidationError、Session 无效、SQLAlchemyError 等异常。

## 十二、三端对照

| 基础设施能力 | dehaze-python | dehaze-java | dehaze-go | 一致性 |
|-------------|---------------|-------------|-----------|--------|
| HTTP 框架 | FastAPI (uvicorn) | Spring MVC (Tomcat) | Gin | 接口语义一致 |
| ORM | SQLAlchemy 2.0 (异步) | MyBatis-Plus | GORM | 功能对等 |
| 缓存 | Redis + local_cache L1 | Spring Cache + 多级 (Caffeine L1 + Redis L2) | 多级缓存 (gkit L1 + Redis) | 已对齐 |
| 消息队列 | aio-pika RabbitMQ (MQ优先+fallback) | RabbitMQ | RabbitMQ | 共享 Exchange/Queue |
| 定时任务 | pyxxl XXL-Job (17 个任务) | @XxlJob (15 个) | XXL-Job (14 个) | 共享 Admin，handler 命名统一 |
| 日志 | Python logging + JSON | Logback | Zap | 格式/级别统一 |
| 认证 | Redis Session | Spring Security + Session | 自研中间件 + Session | Session 机制互通 |
| 权限 | RBAC (Depends + @require_permission) | RBAC (@PreAuthorize) | RBAC (中间件) | 权限标识一致 |
| 错误码 | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | 完全一致 |
| 响应格式 | `{code, msg, data, traceId}` | `{code, msg, data, traceId}` | `{code, msg, data, traceId}` | 完全一致 |
| WebSocket | FastAPI 原生 + Redis Pub/Sub | STOMP + SockJS | - | Python 跨 Worker 已实现 |
| 对象存储 | 多存储后端抽象 (minio/local/nginx-static) | MinIO / 阿里云 OSS | 通过 API | 三端统一 StorageService 抽象 |
| 限流/防重 | 装饰器 (Redis 计数) | 注解 + Redis | 中间件 | 功能对等 |
| IP 黑名单 | 中间件自动封禁 | 中间件 | 中间件 | 三端对齐 |

## 十三、技术栈总览

| 分类 | 技术 | 用途 |
|------|------|------|
| 语言 | Python >= 3.10 | 后端开发 + AI 推理 |
| Web 框架 | FastAPI >= 0.115 | ASGI 异步 HTTP 路由、中间件、WebSocket |
| ASGI 服务器 | uvicorn >= 0.32 | 高性能异步服务器 |
| ORM | SQLAlchemy >= 2.0 | 异步 ORM，声明式模型 |
| 文档数据库 | Motor (MongoDB) >= 3.6 | 异步 MongoDB 客户端，登录/操作审计日志 |
| 缓存 | redis-py (asyncio) >= 6.4 | 异步 Redis 客户端 |
| 对象存储 | MinIO Python SDK >= 7.2 | 文件/图像存储 |
| AI 推理 | PyTorch >= 2.9 + torchvision >= 0.24 | 深度学习推理引擎 |
| 消息队列 | aio-pika (RabbitMQ) >= 9.4 | 异步任务分发（MQ 优先 + asyncio.Task fallback） |
| 定时任务 | pyxxl (XXL-Job) >= 0.4 | 分布式定时任务调度 |
| 配置管理 | pydantic-settings >= 2.0 | 环境变量绑定、多环境配置 |
| 监控 | prometheus-client + starlette-exporter | Prometheus 指标采集 |
| 容器 | Docker (NVIDIA CUDA 12.1) | GPU 推理容器化 |
| 包管理 | uv | 快速 Python 包管理器 |
| 测试 | pytest + pytest-asyncio >= 8.0 | 异步单元/集成测试 |

## 十四、关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | ASGI 异步框架，原生 async/await，自动 OpenAPI 文档 |
| AI 推理 | PyTorch | 深度学习主流框架，30 种算法模型支持 |
| 预测流程 | 责任链模式（拦截器链） | 可插拔预查询/缓存，不修改主流程 |
| 任务分发 | MQ 优先 + asyncio.Task fallback | RabbitMQ 持久化保证 + 进程内降级兜底 |
| 存储 | 策略模式适配多后端 | minio/local/nginx-static 统一抽象 |
| ORM | SQLAlchemy 2.0 异步 | 与 FastAPI 异步模型一致 |
| 配置 | Pydantic Settings v2 | 类型安全，启动期校验，派生地址自动计算 |
| 定时任务 | XXL-Job (pyxxl) | 与 Java/Go 端共享调度中心 |
| WebSocket | 原生 WebSocket + Redis Pub/Sub | 跨 Worker 通信，无第三方依赖 |
| 日志 | Python logging + JSON 结构化 | 按日期分目录，按大小分片，保留 30 天 |
| 监控 | Prometheus（HTTP/GPU/推理/任务四大类指标） | 与 Java/Go 端命名统一 |
| 图像特征分析放 Python 端 | 复用 PyTorch + OpenCV 生态 | 场景分类依赖预训练模型，暗通道/边缘/直方图依赖 OpenCV，Java/Go 端引入这些依赖成本高且无法复用 GPU；Python 端复用已有推理管线和模型缓存 |
| 业务与算法同进程部署 | 共用 FastAPI 分层与基础设施 | 算法密集场景下业务请求常触发推理，同进程避免跨服务 HTTP 往返与模型权重重复加载；代价是 GPU 节点同时承载业务职责，业务侧故障会波及推理——通过异步任务（MQ + TaskTracker）将长耗时推理与同步业务请求解耦，并对推理入口做超时/降级兜底 |
