# 图像去雾系统 (Java 版本)

基于 JDK 17、Spring Boot 3.3、Spring Security 6、JWT、Redis、MyBatis-Plus 构建的图像去雾系统后端。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Java后端架构文档.md)。

## 技术栈

- **框架**: Spring Boot 3.3 + Spring Security 6 (Jakarta EE)
- **数据库**: MySQL 8.4 + MongoDB
- **缓存与存储**: Redis (Redisson) + MinIO / 阿里云 OSS / 本地存储
- **算法集成**: 通过 Python 服务 (dehaze-python) RESTful API 调用
- **安全**: JWT + RBAC + Redis 权限缓存
- **监控**: Prometheus + Grafana + ELK
- **文档**: Knife4j 4.3 (OpenAPI 3)

## 快速开始

1. 执行 [sql/init.sql](sql/init.sql) 初始化数据库
2. 修改 [application-dev.yml](src/main/resources/application-dev.yml) 中的 MySQL、Redis 连接配置
3. 运行 [SystemApplication.java](src/main/java/com/pei/dehaze/SystemApplication.java) 的 main 方法启动

```bash
# 或通过 Maven 启动（跳过测试）
mvn spring-boot:run -DskipTests
```

- 接口文档: `http://localhost:8989/doc.html`
- Python 算法服务地址: `http://127.0.0.1:8991`（配置在 `application-dev.yml`）

## 常用命令

| 命令 | 说明 |
|------|------|
| `mvn spring-boot:run -DskipTests` | 启动服务 |
| `mvn test` | 运行测试 |
| `mvn package -DskipTests` | 打包 |
