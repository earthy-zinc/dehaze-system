# DehazeSystem 项目记忆

## 项目约定

每个项目的启动方式请务必参考项目根目录的 `README.md` 文件，并严格按照其中的说明进行操作。

### 环境变量管理
- 环境变量统一在项目根目录 .env 文件中 `DEHAZE_PASSWORD` 管理所有基础设施密码
- 统一基础设施主机地址 `DEHAZE_HOST`（`.env` 中配置），三端所有基础设施连接（MySQL/Redis/MongoDB/MinIO/RabbitMQ/Nginx/XXL-Job）使用此变量；
- 三个后端（Go/Java/Python）共享相同 JWT 签名密钥，确保 JWT token 可互认

### 调试辅助脚本
- **联调测试工具集**：`dehaze-test/`（替代旧 `scripts/debug_helper.py`）
  - 复用 `dehaze-python/.venv`（含 redis/pymysql/httpx/pytest）
  - `utils/`：config / redis / mysql / auth / api / cleanup（对齐 sdk-js/test/utils/）
  - `tests/`：pytest 集成测试（三端参数化）
  - `scripts/`：login / unread_count / cleanup / compare_backends / db_query / rebuild_mysql
  - 三端后端本机映射端口：java:8989 / go:8990 / python:8991
  - Redis 6379 / MySQL 3306 远程不开放，需 `ssh -L` 转发后直连
  - 三端成功码统一为 `"00000"`（不是 200）