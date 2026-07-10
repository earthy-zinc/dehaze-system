# DehazeSystem 项目记忆

## 项目约定

### 环境变量管理
- 统一密码变量 `DEHAZE_PASSWORD=12345678` 管理所有基础设施密码（MinIO 要求 >= 8 位）
- JWT 签名密钥统一为 `SecretKey012345678901234567890123456789012345678901234567890123456789`，通过 `JWT_SECRET_KEY` 环境变量注入
- 三个后端（Go/Java/Python）共享相同密钥，确保 JWT token 可互认
- `.env` 文件需放置在各后端项目的根目录（Go 加载位置为 CWD，Python/pydantic-settings 同）

### 本地服务连接
- Docker 容器运行于 localhost，端口映射与 docker-compose.yml 一致
- **密码统一为 `12345678`**（所有服务：MySQL/Redis/MongoDB/MinIO/PostgreSQL）
- docker-compose.yml 中基础服务 volumes 已设为 `external: true`，防止重建丢失数据

### Python 项目
- 使用 `uv` 管理依赖（`uv sync`, `uv run`）
- `.venv` 虚拟环境在项目目录下
- 必要环境变量：`SECRET_KEY`, `JWT_SECRET_KEY`, `DEHAZE_PASSWORD`, `MINIO_ACCESS_KEY`

### Java 项目
- Java 17 + Spring Boot 3.3.11 + Maven
- 添加了 `me.paulschwarz:spring-dotenv:4.0.0` 以支持 `.env` 加载
- 跳过测试编译启动：`mvn spring-boot:run -DskipTests -Dmaven.test.skip=true`
