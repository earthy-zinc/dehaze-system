# DehazeSystem 项目记忆

## 用户偏好

### 代码修改原则（必须遵守）
- **禁止兼容历史烂逻辑**：发现新逻辑与历史逻辑矛盾或历史设计不规范时，直接修改为新逻辑，不做无谓的兜底和兼容
- 不保留旧字段名/旧格式/旧接口的向后兼容
- 不写 `or payload.get("old_field")` 之类的 fallback
- 修改字段名/格式时，全局搜索所有引用点一并修改，不要只改一处

## 项目约定

### 环境变量管理
- 统一密码变量 `DEHAZE_PASSWORD=12345678` 管理所有基础设施密码（MinIO 要求 >= 8 位）
- JWT 签名密钥统一为 `SecretKey012345678901234567890123456789012345678901234567890123456789`，通过 `JWT_SECRET_KEY` 环境变量注入
- 三个后端（Go/Java/Python）共享相同密钥，确保 JWT token 可互认
- `.env` 文件需放置在各后端项目的根目录（Go 加载位置为 CWD，Python/pydantic-settings 同）

### 调试辅助脚本（三端 API 一致性验证）
- **登录脚本**：`.codebuddy/scripts/login_helper.py [go|python|java|all]` — 自动获取验证码+登录，返回 token
- **调试脚本**：`.codebuddy/scripts/debug_helper.py <command>` — 封装常用调试操作
  - `status` — 三端服务状态
  - `restart go|python|all` — 重启服务（Go 自动编译）
  - `compare /api/v1/xxx [METHOD] [BODY]` — 三端对比同一 API（自动登录+一致性判断）
  - `curl <backend> /api/v1/xxx [METHOD] [BODY]` — 单端请求
  - `db "SQL"` — MySQL 查询 / `redis get|keys <key> [db]` — Redis 操作
  - `logs python|go` — 服务日志 / `kill <port>` — 杀端口进程
- 账号：admin / 123456（数据库密码也是 123456，基础设施密码是 12345678）
- Git Bash 中 compare 需加 `MSYS_NO_PATHCONV=1` 前缀避免路径转换

### 运行端口记录（勿重复启动）
- **Java**: 8989 (Spring Boot devtools 热重载，已在运行)
- **Go**: 8990 (二进制，改代码后需 `go build` + 重启)
- **Python**: 8991 (启动 `.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8991`)
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
- Go 的 `config.yaml` 和 Python 的 `config.py` 中所有 host 均已改为 `127.0.0.1`
- Python `DevelopmentSettings.XXLJOB_ENABLED=False`（Docker 中未运行 xxl-job-admin）
- 后台进程在 Git Bash 间不持久，Python 需在独立终端启动

### Python 项目
- 使用 `uv` 管理依赖（`uv sync`, `uv run`）
- `.venv` 虚拟环境在项目目录下
- 必要环境变量：`SECRET_KEY`, `JWT_SECRET_KEY`, `DEHAZE_PASSWORD`, `MINIO_ACCESS_KEY`

### Java 项目
- Java 17 + Spring Boot 3.3.11 + Maven
- 添加了 `me.paulschwarz:spring-dotenv:4.0.0` 以支持 `.env` 加载
- 跳过测试编译启动：`mvn spring-boot:run -DskipTests -Dmaven.test.skip=true`
- **权限校验**：`@PreAuthorize("@ss.hasPerm('xxx')")` 通过 `PermissionService.hasPerm()` 校验，ROOT 角色绕过所有检查（`SecurityUtils.isRoot()`）
- admin 用户已关联 ROOT + ADMIN 角色，可通过所有权限校验
- **Python 算法服务 URL**：`http://127.0.0.1:8991`（不是 5000），配置在 `AlgorithmProperties` + `application-dev.yml`

### dehaze-uniapp 依赖版本约束（重要）
- **Pinia 必须锁定 `2.2.4`**（精确版本，不能用 `^` 或 `~`）：uni-app 的 `@dcloudio/uni-h5` 硬编码依赖 `vue: 3.4.21`，而 pinia 2.2.5+ / 3.x 要求 `vue ^3.5.11`，不兼容
- **`@vue/devtools-api: ^6.6.4` 必须作为直接依赖显式声明**：pnpm 不会把 transitive dep 提升到顶层 `node_modules`，导致 Vite 无法解析 pinia 中的 `import "@vue/devtools-api"`
- **@dcloudio 主包版本**：使用 npm `vue3` dist-tag 对应的版本（非 `latest` tag，后者是 vue2 版本）。所有 @dcloudio/* 包共享同一版本号
- **uvm 工具**：`npx @dcloudio/uvm@latest` 强制交互模式，无法非交互运行，需手动查 registry 更新 package.json

### dehaze_harmory 鸿蒙应用
- 基于 OpenHarmony/HarmonyOS（ArkTS + ArkUI），SDK 6.0.0(20)
- 依赖 `dehaze-sdk-harmony`（file:../../dehaze-tool/dehaze-sdk-harmony）
- SDK 核心已修复：HttpManager 用 `@kit.NetworkKit`、TokenManager 用 `@kit.ArkData`、Result 用 `code: string + msg`
- 应用页面：Login → Home → ImageInput → AlgorithmSelect → Processing → Compare
- 后端地址：`http://127.0.0.1:8989`（EntryAbility 中配置）
- EntryAbility.onCreate 需 `await DehazeSDK.initialize(context, builder)` + `await TokenManager.initialize(context)`

### SDK 测试（dehaze-tool/dehaze-sdk-js）
- vitest 集成测试需在 `vitest.setup.ts` 中设置 `javaService.defaults.baseURL`（Node.js 无浏览器 origin）
- 登录需先获取验证码 → 从 Redis db0 读取 captcha code（Jackson 序列化带双引号需去除）
- 响应拦截器已支持 `arraybuffer` + `blob` 两种二进制响应类型
- 预测/评估正向测试依赖 Python 服务 + 真实图片 + 模型文件，基础设施未就绪时优雅 skip
