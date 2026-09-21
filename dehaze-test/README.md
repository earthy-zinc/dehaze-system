# dehaze-test

**开发工作台（Developer Workbench）**：面向"人的开发活动"的工具集——调试、数据操作、跨端对比、质量评估。
不是产品代码的测试场所。

## 定位与边界

与三个邻居的职责边界（防止边界腐烂的核心约束）：

| | 承载 | 触发 | 产物 | 判定 |
|---|---|---|---|---|
| `dehaze-python/tests` 等各端 tests | 产品代码回归（断言正确性） | 每次改动 / CI | pass/fail | **精确预期值** |
| `dehaze-sdk-js/test` | SDK ↔ 后端契约集成 | 发版 / 契约变更 | pass/fail | 契约一致 |
| **dehaze-test（本项目）** | 开发过程辅助 | **人工按需** | 数据 / 报告 / 环境状态 | **无预期值，供人决策** |

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
4. **工具复用**：`utils/` 与 `dehaze-sdk-js/test/utils/` 保持设计对齐（config/redis/mysql/auth/api/cleanup）

## 目录

```
dehaze-test/
├── utils/         # 工具库（config/redis/mysql/auth/api/cleanup/sse）
├── tests/         # 三端对比集成测试（pytest，人工触发，非 CI）
├── scripts/       # 联调调试 + 环境数据操作脚本
└── kb_eval/       # 知识库质量评估（分块质量 + 召回质量，报告归档 reports/）
```

## 运行环境

复用 `dehaze-python` 的 venv（已含 `redis 6.4` / `pymysql 1.4` / `httpx 0.28` / `pytest 8.4`）：

```bash
PYTHON=/data/workspace/dehaze-system/dehaze-python/.venv/bin/python
```

## 网络前提

三端后端通过本机映射端口访问（Java:8989 / Go:8990 / Python:8991），开箱即用。

**Redis（6379）和 MySQL（3306）默认指向 `MYSQL_HOST` / `REDIS_HOST`（`127.0.0.1`），远程端口可能不开放**。需要直连时（如查未读消息数、清理缓存、重建数据库），自行做端口转发：

```bash
ssh -L 6379:127.0.0.1:6379 -L 3306:127.0.0.1:3306 <user>@<MYSQL_HOST>
```

转发完成后，dehaze-test 会通过 `MYSQL_HOST` / `REDIS_HOST`（被 ssh 转发到 127.0.0.1）直连 Redis/MySQL。

## 使用

### 跑三端对比集成测试

```bash
cd dehaze-test
../dehaze-python/.venv/bin/python -m pytest tests/ -v
```

### 跑 ad-hoc 脚本

```bash
# 登录获取 session
../dehaze-python/.venv/bin/python scripts/login.py --backend java --user admin

# 查未读消息数（API + DB 双重验证）
../dehaze-python/.venv/bin/python scripts/unread_count.py --user admin

# 三端 API 响应对比
../dehaze-python/.venv/bin/python scripts/compare_backends.py /api/v1/auth/captcha

# 交互式 SQL 查询
../dehaze-python/.venv/bin/python scripts/db_query.py "SELECT COUNT(*) FROM sys_message"

# 重建数据库（危险操作：需显式 --only 指定库名；--only dehaze 会清空开发库全部数据）
../dehaze-python/.venv/bin/python scripts/rebuild_mysql.py --only dehaze
```

### 知识库质量评估

```bash
# 离线分块评估（纯算法，不起后端，秒级）
../dehaze-python/.venv/bin/python -m kb_eval.offline_eval

# 检索链路评估（需 8991 已启动，走真实链路）
../dehaze-python/.venv/bin/python -m kb_eval.retrieval_eval
```

详见 `kb_eval/README.md`。

### 在自己的脚本中复用工具库

```python
import sys
sys.path.insert(0, "/data/workspace/dehaze-system/dehaze-test")

from utils import auth, mysql, redis, api

sid = auth.login("admin", backend="java")
resp = api.get("/api/v1/messages/unread-count", backend="java")
print("API 未读数:", resp["data"]["count"])
```

## 配置

从项目根 `.env` 读取：

- `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USERNAME` / `MYSQL_PASSWORD` / `MYSQL_DATABASE`：MySQL 直连配置
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_DATABASE`：Redis 直连配置
- `ADMIN_PASSWORD`：登录种子账号 admin 的密码（基础设施密码统一）

三端后端固定映射到本机端口（与 `dehaze-sdk-js/test/config/constant.ts` 一致）：

| 后端 | 端口 |
|---|---|
| dehaze-java | 8989 |
| dehaze-go | 8990 |
| dehaze-python | 8991 |
