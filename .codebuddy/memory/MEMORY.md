# DehazeSystem 项目记忆

## 项目约定

每个项目的启动方式请务必参考项目根目录的 `README.md` 文件，并严格按照其中的说明进行操作。

### 协作方式（2026-08-24 用户明确要求）
- **先分析后动手**：涉及设计层面疑问/重大设计决策（架构、数据模型、接口、命名、可扩展性方向等），必须先给出分析与方案对比，**经用户确认后再修改文档/代码**，不得未经同意直接改动
- 用户认可的设计决策示例：用户身份透传配置采用 `sys_ai_provider.user_identity_forward`（JSON 单字段：enabled/field/prefix/max_len，抽象覆盖 DeepSeek user_id / OpenAI user / Anthropic metadata.user_id 差异）

### 模块架构约定（2026-08-25）
- **AI 模型管理为独立基础模块**（`03-模块设计/基础模块/AI模型管理/`，需求/API/后端/测试/前端五文档）：原 AI 对话 F-M08-007 模型管理功能域抽离（因被 AI 对话 + AI 知识库多模块消费）；供应商管理（API Key/健康容错/限速隔离/用户身份透传）+ 模型注册表（`model_type`=chat/embedding/rerank）+ 生命周期统一承载
- **模型类型扩展**：`sys_ai_model` 含 `model_type` + `dimension`（embedding 向量维度，创建后不可改，知识库 ES 索引映射依赖）；知识库 embedding/rerank 从注册表选择（`GET /api/v1/ai/models/enabled?model_type=`），对话消费 `model_type=chat`；模型管理页路由 `/admin/ai-models`（`ai:model:manage`）
- 跨模块文档引用路径：核心模块→基础模块 `../../基础模块/`、同级模块 `../`、docs 根 `../../../02-系统架构/`；cp 迁移文档后必须修正旧相对路径

### AI对话模块文档结构（2026-08-28 目录级重组 + 新增两功能域）
- **保持单一模块**（各功能域共享同一推理运行时，不满足"被多模块消费才抽离"判据），仅做目录级重组：`核心模块/AI对话/` 根目录保留 4 个总览/公共文档（需求规格/API接口/测试用例/后端实现-架构与公共）+ 11 个功能域子目录（会话与消息/多步推理/上下文管理/智能调度/能力扩展/消息反馈/智能体管理/定时与兼容/**可观测性**/**Agent评测**/页面设计）
- 子目录内统一命名 `需求规格.md`/`后端实现.md`/`测试用例.md`/`前端实现.md`；例外：`定时与兼容/` 有两篇后端（`后端实现-定时调度.md`、`后端实现-第三方兼容.md`），`页面设计/` 保留 `前端实现-*` 原名
- **可观测性（F-M08-013，章节 2.12）**：自建方案，双粒度观测——消息级 `sys_ai_trace`（含 first_token_ms/cached_tokens/llm_call_count/context_snapshot）+ 每次 LLM 调用 span 级 `sys_ai_llm_call`（trace_id+seq 唯一、step_position 关联 thought）；管理端「AI 可观测中心」页（`/admin/ai-observability`，`ai:conversation:audit`）
- **Agent 评测（F-M08-014，章节 2.13）**：自研评测器，复用 `sys_ai_agent_eval_dataset/sample/run` 三表；发布门禁触发留在智能体管理，评测实现收敛至评测域；管理端「评测中心」页（`/admin/ai-eval-center`，`ai:agent:manage`）
- 新增功能域文档时按此结构落位；引用路径规则同上但需注意子目录多一层（如域内文档→基础模块为 `../../../基础模块/`）

### AI对话推理实现事实（2026-08-28 文档治理时以代码为准确认）
- **推理循环已改用 deepagents 内部循环**：手动 `react_loop`/`execute_tools` 节点已废弃（dehaze-python 无此代码），工具装载走 `before_model` 钩子、横切逻辑走 Hooks；文档已全量同步
- **checkpoint 运行时层**：`RedisSaver` 为自定义实现（langgraph 1.2.x 未内置，继承 BaseCheckpointSaver，`app/infrastructure/cache/checkpoint_manager.py`）
- **`MySQLCheckpointSaver` 三端代码均未实现**（dehaze-python/java/go 均无），但公共文档 §4.3、多步推理 §2.2、智能体管理 §3.5、数据库设计 §4.7 均声称存在 Redis+MySQL 双层——若持久化层仍规划中需明确，已废弃需删（2026-08-28 已报告待用户决策，未擅改设计描述）
- `ai:model:list` 缓存 TTL=1 小时（`MODEL_LIST_CACHE_TTL = CACHE_TTL_HOUR`）；代码中无 `ai:model:{modelId}` 单模型缓存

### MCP 双通道术语约定（2026-08-28 概念治理，单一信息源在 `核心模块/MCP/需求规格.md` §1.0）
- **内部 MCP 能力网关**（dehaze-mcp-gateway，自建 MCP Server）＝系统能力出口：启动时一次性读取后端 OpenAPI（**无定时同步**，新增 API 需重启网关生效），3 个元 tool；dehaze-python 经 `McpGatewayClient` 连接
- **MCP Server 管理**＝外部能力进口（AI 对话能力扩展域）：任意符合 MCP 协议的第三方服务注册接入（sys_ai_mcp_* 四表、stdio/streamable-http/sse、凭据 AES、健康探测）；`McpToolFetcher` 拉取工具清单
- 术语不可混用：文档写"MCP 网关"一律指内部能力网关；dehaze-python 在两条通道中都是 MCP 客户端
- **待用户决策的不一致**：能力扩展 §5/需求规格 §2.6.13 声称外部 Server 工具"统一纳入命名空间预筛选与工具装载"，但 `DehazeToolsBuilder` 仅装载内部网关元工具、prefilter 无外部 server 逻辑——外部工具运行时装载未实现

### 环境变量管理（2026-08-23 重构：按基础设施分区）
- 项目根 `.env` 按基础设施分区，各服务 HOST/PORT/凭证独立变量：`MYSQL_*` / `REDIS_*` / `MONGODB_*` / `ES_*` / `MINIO_*` / `RABBITMQ_*` / `XXLJOB_*` / `NGINX_STATIC_*` / `GRAFANA_ADMIN_PASSWORD` / `ALERTMANAGER_PASSWORD`；应用级：`JWT_SECRET_KEY`、`DEFAULT_PASSWORD`（新用户初始密码，三端统一）、`ADMIN_PASSWORD`（种子账号 admin 登录密码，bcrypt 固化在 sys_user.sql，改 .env 不会改账号真实密码）
- 旧统一变量 `DEHAZE_HOST`/`DEHAZE_PASSWORD` 已彻底删除（零回退），三端占位符同步替换完毕
- 三端加载机制：Java spring-dotenv / Go godotenv+os.ExpandEnv（viper.go）/ Python pydantic-settings；pydantic 中 dotenv 优先级高于类默认值，测试库隔离靠 conftest 在导入 app 前强制 os.environ 覆盖（MYSQL_DATABASE=dehaze_test、REDIS_PORT=6390）
- 三个后端（Go/Java/Python）共享相同 JWT 签名密钥，确保 JWT token 可互认

### 调试辅助脚本
- **联调测试工具集**：`dehaze-test/`
  - 复用 `dehaze-python/.venv`（含 redis/pymysql/httpx/pytest）
  - `utils/`：config / redis / mysql / auth / api / cleanup（对齐 sdk-js/test/utils/）
  - `tests/`：pytest 集成测试（三端参数化）
  - `scripts/`：login / unread_count / cleanup / compare_backends / db_query / rebuild_mysql
  - 三端后端本机映射端口：java:8989 / go:8990 / python:8991
  - Redis 6379 / MySQL 3306 远程不开放，需 `ssh -L` 转发后直连
  - 三端成功码统一为 `"00000"`（不是 200）

### 数据库迁移机制（2026-08-26 用户确认）
- **pytest 测试库**：conftest `_mysql_schema`（session 级）每次运行自动 DROP+CREATE 重建 `dehaze_test`，从 `config/sql/schema/*.sql`+`data/*.sql` 全量导入——测试库无需手动迁移，但 config/sql 下 SQL 必须最新且可执行（新表 SQL 放入 `config/sql/schema/` 自动生效；SQL 语法错会导致测试库重建失败、所有 db 用例全挂）
- **SDK 集成测试（pnpm test:python）连的是开发库 `dehaze`（非测试库）**：跑之前必须先迁移开发库表结构（增量执行新增表/字段 SQL，或 `python dehaze-test/scripts/rebuild_mysql.py --only dehaze`，后者会清空 dehaze 全部数据，慎用，需 root 直连 MySQL）

### dehaze-test 使用规范（2026-08-26 用户明确要求）
- dehaze-test 是**辅助开发测试脚本集**，必须按用途正确调用（rebuild_mysql 重建库/清空数据慎用、db_query 只读查库、login/cleanup 等），禁止猜测式调用
- 若发现脚本不足以支撑任务（缺迁移/状态检查能力），应**优化脚本本身**而非重复调用/绕过
- 避免反复重复运行测试命令；跑 SDK 测试前先确认：服务是否已起（scripts/run.py）、开发库表结构是否已迁移、Redis 是否需清理，一步到位

### 配置项设计规范（2026-08-28 用户确认，全模块已落地）
- 判定标准：仅"管理员/运营有明确修改动机"的参数可配置（sys_dict 或应用配置/环境变量）；纯技术参数（缓存TTL/分页/批次/轮询/锁定/心跳/cron/线程池）一律代码常量；个性化由实体级配置（Agent/会话/知识库库级）覆盖，**不设第三层全局字典默认值**
- AI对话推理参数默认值为代码常量 `agent_config_resolver.REASONING_DEFAULTS`（ai_reasoning_defaults 字典已删除，勿再引用）；护栏默认 `ai_guardrail_defaults` 保留 sys_dict（安全合规有运维场景）
- 字典化业务参数：`favorite_capacity`（default/vip1/vip2/svip=200/500/1000/3000）、`member_growth_rules`（sign_in_value=3/sign_in_streak_bonus=20/rating_growth_value=5/rating_growth_daily_limit=5），三端消费方均带缓存（TTL 1h）+缺键 warn 回退设计默认值；python 侧入口 `dict_service.get_dict_int`/`ensure_system_dict_defaults`
- 新增 sys_dict 种子需同步：config/sql/data/*.sql + python dict_service 种子 + SYSTEM_PRESET_DICT_TYPE_CODES + seed contract 测试