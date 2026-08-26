# dehaze-test

跨多端（Java / Go / Python 后端 + 各前端）联调与集成测试工具集，对齐
`dehaze-sdk-js/test` 的设计哲学：**模块化工具函数 + pytest 用例**。

## 设计目标

- 三端后端 API 对比与回归
- 联调时的 ad-hoc 调试脚本（登录、查 DB、查 Redis、验证业务状态）
- 直连 MySQL / Redis（不依赖本地 docker）
- 多后端、多用户登录支持

## 运行环境

复用 `dehaze-python` 的 venv（已含 `redis 6.4` / `pymysql 1.4` / `httpx 0.28` / `pytest 8.4`）：

```bash
PYTHON=/Users/earthywu/Projects/dehaze-system/dehaze-python/.venv/bin/python
```

## 网络前提

三端后端通过本机映射端口访问（Java:8989 / Go:8990 / Python:8991），开箱即用。

**Redis（6379）和 MySQL（3306）默认指向 `MYSQL_HOST` / `REDIS_HOST`（`127.0.0.1`），远程端口可能不开放**。需要直连时（如查未读消息数、清理缓存、重建数据库），自行做端口转发：

```bash
ssh -L 6379:127.0.0.1:6379 -L 3306:127.0.0.1:3306 <user>@<MYSQL_HOST>
```

转发完成后，dehaze-test 会通过 `MYSQL_HOST` / `REDIS_HOST`（被 ssh 转发到 127.0.0.1）直连 Redis/MySQL。

## 目录

```
dehaze-test/
├── utils/         # 工具库（config/redis/mysql/auth/api/cleanup）
├── tests/         # pytest 集成测试
└── scripts/       # ad-hoc 调试脚本
```

## 使用

### 跑 pytest 集成测试

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

# 重建数据库（迁移自 scripts/rebuild_mysql.sh，去 docker 依赖）
../dehaze-python/.venv/bin/python scripts/rebuild_mysql.py
```

### 在自己的脚本中复用工具库

```python
import sys
sys.path.insert(0, "/Users/earthywu/Projects/dehaze-system/dehaze-test")

from utils import auth, mysql, redis, api

sid = auth.login("admin", backend="java")
resp = api.get("/api/v1/messages/unread-count", backend="java")
print("API 未读数:", resp["data"]["count"])

print("DB 未读数:", mysql.query_one(
    "SELECT COUNT(*) AS cnt FROM sys_message WHERE read_status = 0 AND create_by = %s",
    (1,)
)["cnt"])
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
