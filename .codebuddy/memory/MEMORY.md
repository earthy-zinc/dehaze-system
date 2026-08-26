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