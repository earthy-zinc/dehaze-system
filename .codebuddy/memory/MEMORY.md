# DehazeSystem 项目记忆

## 项目约定

### 环境变量管理
- 统一密码环境变量 `DEHAZE_PASSWORD=12345678` 管理所有基础设施密码
- JWT 签名密钥统一为 `JWT_SECRET_KEY=SecretKey012345678901234567890123456789012345678901234567890123456789`，三个后端（Go/Java/Python）共享相同密钥，确保 JWT token 可互认
- `.env` 文件需放置在各后端项目的根目录

### API 字段命名约定
- 获取当前用户信息统一用 `GET /api/v1/auth/me`（三端一致，已废弃 `/users/me`）
- 响应 JSON 权限字段名为 `perms`（不是 `permissions`），三端 + 所有 SDK 一致

### 调试辅助脚本
- **调试脚本**：`scripts/debug_helper.py <command>` - 包含登录逻辑 + API 调试
  - `compare /api/v1/xxx [METHOD] [BODY]` - 三端对比同一 API（自动登录+一致性判断）
  - `curl <backend> /api/v1/xxx [METHOD] [BODY]` - 单端请求
- 账号：admin / 123456（基础设施密码是 12345678）

### 后端生命周期管理
- **统一脚本**：`scripts/run.py <command> [args...]`（103 行，极简版）
- 命令：`run|stop|restart <svc>[,svc...]|all`、`ps`、`logs <svc> [lines]`
- 开发模式启动：`go run ./cmd/main.go` / `uvicorn --reload --host 0.0.0.0 --port 8991` / `mvn spring-boot:run -DskipTests`
- 支持别名（`go`/`python`/`java`）和 `all`，PID 文件落各服务目录（`.<svc>.pid`）
- 无端口探测、无 netstat 解析，纯 PID 文件跟踪

### 运行端口记录

以下服务已经启动，请勿重复启动，如需重启，请告知用户，切勿私自重启

- **Java**: 8989 (Spring Boot devtools 热重载)
- **Go**: 8990 (`go run` 开发模式)
- **Python**: 8991 (`uvicorn --reload` 开发模式)
- Docker 容器: MySQL 3306, Redis 6379, MongoDB 27017, MinIO 9000/9090, PostgreSQL 5432, RabbitMQ 5672/15672

### JWT claims 统一格式（三端互认）
三端 JWT token 可互认，统一 claims 结构（以 Java/Go 为标准）：
- `jti`: UUID (token ID)
- `sub`: username (用户名)
- `userId`: int (用户ID，camelCase)
- `authorities`: 数组 (角色，元素带 `ROLE_` 前缀，如 `["ROLE_GUEST"]`)
- `deptId`: int (部门ID，Java/Go 有，Python 暂未生成)
- `dataScope`: int (数据权限，Java/Go 有，Python 暂未生成)
- `exp`/`iat`: 时间戳

Python 额外包含 `username`/`nickname`/`permissions`/`type` 字段（不影响互认）。
Java captcha 存 Redis db0（Jackson 序列化带引号），Go/Python captcha 存 db3（纯文本）。

### Windows 开发环境注意事项
- **必须用 `127.0.0.1` 而非 `localhost`**：Windows 上 `localhost` 解析为 IPv6 `::1`（优先）+ IPv4 `127.0.0.1`。Docker Desktop 端口映射只绑定 IPv4。Python `socket.create_connection` 串行尝试地址，先连 `::1` 失败后等 TCP SYN 超时（约 21 秒）才回退 IPv4。Go 不受影响（Happy Eyeballs 并行尝试），Java 也不受影响（默认偏好 IPv4）

### Python 项目
- 使用 `uv` 管理依赖（`uv sync`, `uv run`）
- `.venv` 虚拟环境在项目目录下
- 必要环境变量：`SECRET_KEY`, `JWT_SECRET_KEY`, `DEHAZE_PASSWORD`, `MINIO_ACCESS_KEY`

### Java 项目
- Java 17 + Spring Boot 3.3.11 + Maven
- 跳过测试编译启动：`mvn spring-boot:run -DskipTests -Dmaven.test.skip=true`
- **权限校验**：`@PreAuthorize("@ss.hasPerm('xxx')")` 通过 `PermissionService.hasPerm()` 校验，ROOT 角色绕过所有检查（`SecurityUtils.isRoot()`）
- admin 用户已关联 ROOT + ADMIN 角色，可通过所有权限校验

### dehaze_harmory 鸿蒙应用
- 基于 OpenHarmony/HarmonyOS（ArkTS + ArkUI），SDK 6.0.0(20)
- 依赖 `dehaze-sdk-harmony`（file:../../dehaze-tool/dehaze-sdk-harmony）

### SDK 测试（dehaze-tool/dehaze-sdk-js）
- vitest 集成测试需在 `vitest.setup.ts` 中设置 `javaService.defaults.baseURL`（Node.js 无浏览器 origin）
- 登录需先获取验证码 -> 从 Redis db0 读取 captcha code（Jackson 序列化带双引号需去除）
- 响应拦截器已支持 `arraybuffer` + `blob` 两种二进制响应类型
