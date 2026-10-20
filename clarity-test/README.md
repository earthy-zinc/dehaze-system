# clarity-test

**开发工作台（Developer Workbench）**：面向"人的开发活动"的工具集——调试、数据操作、跨端对比、质量评估。
不是产品代码的测试场所。

## 定位与边界

与三个邻居的职责边界（防止边界腐烂的核心约束）：

|                                    | 承载                       | 触发            | 产物                   | 判定                   |
| ---------------------------------- | -------------------------- | --------------- | ---------------------- | ---------------------- |
| `clarity-python/tests` 等各端 tests | 产品代码回归（断言正确性） | 每次改动 / CI   | pass/fail              | **精确预期值**         |
| `clarity-sdk-js/test`               | SDK ↔ 后端契约集成         | 发版 / 契约变更 | pass/fail              | 契约一致               |
| **clarity-test（本项目）**          | 开发过程辅助               | **人工按需**    | 数据 / 报告 / 环境状态 | **无预期值，供人决策** |

四类任务：

1. **联调调试**：登录态获取、DB/Redis 查询、键清理、业务状态验证（ad-hoc，scripts/）
2. **环境与数据管理**：开发库重建、增量迁移、种子校验（危险操作集中管控，scripts/）
3. **跨端对比**：三端同接口行为对比，暴露契约漂移（scripts/ + tests/）
4. **质量评估**：度量型任务（如 kb_eval），产出指标报表与版本化基线——**评估发现的硬性不变量必须沉淀回各端 pytest**，不许留在这里当长期断言

**不承载**：任何进 CI 的东西、产品代码的回归测试、SDK 契约测试。

## 治理规则

1. **危险操作白名单化**：`rebuild_mysql` / `cleanup` / 迁移类脚本必须显式指定目标（库名 / 键前缀），禁止无参默认全量
2. **一次性脚本生命周期**：用完即删或移入 `archive/`，不长期滞留
3. **评估产物归档**：`kb_eval/reports/` 版本化基线，报告可追溯
4. **工具复用**：`utils/` 与 `clarity-sdk-js/test/utils/` 保持设计对齐（config/redis/mysql/auth/api/cleanup）

## 目录

```
clarity-test/
├── utils/         # 工具库（config/redis/mysql/auth/api/cleanup/sse）
├── tests/         # 三端对比集成测试（pytest，人工触发，非 CI）
├── scripts/       # 联调调试 + 环境数据操作脚本
└── kb_eval/       # 知识库质量评估（分块质量 + 召回质量，报告归档 reports/）
```

## 运行环境

复用 `clarity-python` 的 venv（已含 `redis 6.4` / `pymysql 1.4` / `httpx 0.28` / `pytest 8.4`）：

```bash
PYTHON=../clarity-python/.venv/bin/python   # 在仓库根的 clarity-test/ 目录下执行
```

## 网络前提

三端后端通过本机映射端口访问（Java:8989 / Go:8990 / Python:8991），开箱即用。

MySQL / Redis 等基础设施的连接参数一律取自仓库根 `.env`（`MYSQL_*` / `REDIS_*`，与 `clarity-sdk-js/test/config/constant.ts` 同源，SDK 侧另读 `NGINX_STATIC_*`），代码里不写死地址、**缺失即报错**（不再回落到 `127.0.0.1` 或生产预留库名）；这些变量同时决定库名与 Redis db（dev `dehaze_dev` + db1 / pre `dehaze_pre` + db2），**换环境就是换 `.env`**。

dev（21.91.16.245）与 pre（21.6.51.87）的 `.env` 均把 `*_HOST` 指向中间件宿主，在两台机器上直连即可。从办公网工作站跑同一套工具时同样可直连：中间件宿主的 Redis（6381）、nginx 静态（19000）、MySQL（3306）、Mongo（27017）等端口均不在链路侧代理的封禁名单内（被封禁的是 6379 / 9000 等默认端口，客户端表现为 502，不是服务故障），因此**不需要端口转发**。

## 使用

### 跑三端对比集成测试

```bash
cd clarity-test
../clarity-python/.venv/bin/python -m pytest tests/ -v
```

### 跑 ad-hoc 脚本

```bash
# 登录获取 session
../clarity-python/.venv/bin/python scripts/login.py --backend java --user admin

# 查未读消息数（API + DB 双重验证）
../clarity-python/.venv/bin/python scripts/unread_count.py --user admin

# 三端 API 响应对比
../clarity-python/.venv/bin/python scripts/compare_backends.py /api/v1/auth/captcha

# 交互式 SQL 查询
../clarity-python/.venv/bin/python scripts/db_query.py "SELECT COUNT(*) FROM sys_message"

# 重建数据库（危险操作：需显式 --only 指定库名；--only clarity 会清空开发库全部数据）
../clarity-python/.venv/bin/python scripts/rebuild_mysql.py --only clarity

# 环境重置（把 .env 指向的命名空间整体清空并回到种子态：MySQL + Redis + MinIO + MongoDB + ES；
# 破坏性操作须 --target 等于 .env 的 MYSQL_DATABASE 且显式 --yes，先 --dry-run 可只打印计划）
../clarity-python/.venv/bin/python scripts/reset_env.py --target dehaze_pre --dry-run
../clarity-python/.venv/bin/python scripts/reset_env.py --target dehaze_pre --yes
```

### 知识库质量评估

```bash
# 离线分块评估（纯算法，不起后端，秒级）
../clarity-python/.venv/bin/python -m kb_eval.offline_eval

# 检索链路评估（需 8991 已启动，走真实链路）
../clarity-python/.venv/bin/python -m kb_eval.retrieval_eval
```

详见 `kb_eval/README.md`。

### 在自己的脚本中复用工具库

```python
import sys
sys.path.insert(0, "<仓库根>/clarity-test")

from utils import auth, mysql, redis, api

sid = auth.login("admin", backend="java")
resp = api.get("/api/v1/messages/unread-count", backend="java")
print("API 未读数:", resp["data"]["count"])
```

## 配置

从项目根 `.env` 读取：

- `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USERNAME` / `MYSQL_PASSWORD` / `MYSQL_DATABASE`：MySQL 直连配置
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_DATABASE`：Redis 直连配置

三端后端固定映射到本机端口（与 `clarity-sdk-js/test/config/constant.ts` 一致）：

| 后端          | 端口 |
| ------------- | ---- |
| clarity-java   | 8989 |
| clarity-go     | 8990 |
| clarity-python | 8991 |
