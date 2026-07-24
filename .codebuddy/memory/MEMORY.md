# DehazeSystem 项目记忆

## 项目约定

每个项目的启动方式请务必参考项目根目录的 `README.md` 文件，并严格按照其中的说明进行操作。

### Windows 开发环境注意事项
- **必须用 `127.0.0.1` 而非 `localhost`**：
- Windows 上 `localhost` 解析为 IPv6 `::1`（优先）+ IPv4 `127.0.0.1`。
- Docker Desktop 端口映射只绑定 IPv4。
- Python `socket.create_connection` 串行尝试地址，先连 `::1` 失败后等 TCP SYN 超时（约 21 秒）才回退 IPv4。
- Go 不受影响（Happy Eyeballs 并行尝试）
- Java 也不受影响（默认偏好 IPv4）

### 环境变量管理
- 统一密码环境变量 `DEHAZE_PASSWORD=12345678` 管理所有基础设施密码
- 三个后端（Go/Java/Python）共享相同 JWT 签名密钥，确保 JWT token 可互认
- `.env` 文件位于 monorepo 根目录 `DehazeSystem/.env`

### 调试辅助脚本
- **调试脚本**：`scripts/debug_helper.py <command>` - 包含登录逻辑 + API 调试
  - `compare /api/v1/xxx [METHOD] [BODY]` - 三端对比同一 API 确保一致性
  - `curl <backend> /api/v1/xxx [METHOD] [BODY]` - 单端请求
- 开发账号：admin / 123456

### 后端生命周期管理
- **统一脚本**：`scripts/run.py <command> [args...]`
- 命令：`run|stop|restart <svc>[,svc...]|all`、`ps`、`logs <svc> [lines]`
- 支持别名（`go`/`python`/`java`）和 `all`，PID 文件落各服务目录（`.<svc>.pid`）

### pnpm workspace 依赖链接
- **必须用 `workspace:*` 而非 `link:`** 链接 workspace 内部包（如 `dehaze-sdk-js`）
- `link:` 只创建裸 symlink，不走 pnpm peer 依赖解析，导致 TypeScript `exports` 类型解析失败（报 7016/2305 错误）
- `workspace:*` 正确解析 peer 依赖，TypeScript 能正常加载 `.d.ts` 类型声明
- `pnpm-workspace.yaml` 已声明所有子项目为 workspace 成员

以下服务已经启动，请勿重复启动，如需重启，请告知用户，切勿私自重启

- **Java**: 8989 (Spring Boot devtools 热重载)
- **Go**: 8990 (`go run` 开发模式)
- **Python**: 8991 (`uvicorn --reload` 开发模式)
- Docker 容器: MySQL 3306, Redis 6379, MongoDB 27017, MinIO 9110(API)/9190(Console), PostgreSQL 5432, RabbitMQ 5672/15672
