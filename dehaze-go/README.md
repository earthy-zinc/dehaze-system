# 图像去雾系统 (Go 版本)

基于 Go 1.25、Gin、GORM、JWT、Redis 构建的图像去雾系统后端。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Go后端架构文档.md)。

## 技术栈

- **框架**: Gin + GORM
- **数据库**: MySQL 8.4 + MongoDB + Redis
- **存储**: MinIO / 阿里云 OSS / 本地存储
- **安全**: JWT + RBAC + Redis 权限缓存
- **监控**: Prometheus + Grafana
- **文档**: Swagger

## 快速开始

1. 确保 MySQL 8.4+、Redis 6.0+ 已启动
2. 修改 `config/config.yaml` 配置数据库连接等信息
3. 在项目根目录创建 `.env` 文件（设置 `DEHAZE_PASSWORD` 等环境变量）
4. 启动服务：

```bash
# 推荐方式（自动编译、端口检查、后台启动）
./start.sh

# 或直接运行
go run ./cmd/main.go
```

- 接口文档: `http://localhost:8990/swagger/index.html`
- 日志: `log/dehaze-go.log`

## 常用命令

| 命令 | 说明 |
|------|------|
| `./start.sh` | 一键启动 |
| `go run ./cmd/main.go` | 直接运行 |
| `go test ./...` | 运行测试 |
| `go generate ./cmd` | 生成 Mock 代码 |
| `go build ./cmd/main.go` | 编译二进制 |

## 数据库初始化

```bash
mysql -u root -p < ./sql/schema.sql
```
