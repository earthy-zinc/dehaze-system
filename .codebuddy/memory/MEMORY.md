# DehazeSystem 项目记忆

## 项目约定

每个项目的启动方式请务必参考项目根目录的 `README.md` 文件，并严格按照其中的说明进行操作。

### 环境变量管理
- 环境变量统一在项目根目录 .env 文件中 `DEHAZE_PASSWORD` 管理所有基础设施密码
- 统一基础设施主机地址 `DEHAZE_HOST`（`.env` 中配置），三端所有基础设施连接（MySQL/Redis/MongoDB/MinIO/RabbitMQ/Nginx/XXL-Job）使用此变量；
- 三个后端（Go/Java/Python）共享相同 JWT 签名密钥，确保 JWT token 可互认

### 调试辅助脚本
- **调试脚本**：`scripts/debug_helper.py <command>` - 包含登录逻辑 + API 调试