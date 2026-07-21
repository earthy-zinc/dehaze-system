# 图像去雾系统（微服务增强版）

基于 Spring Cloud 2024 + Spring Boot 3.4 + Java 17 构建的分布式图像去雾系统，集成 34 种主流去雾算法。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Java微服务增强版架构文档.md)。

## 技术栈

- **微服务框架**: Spring Cloud 2024 + Spring Boot 3.4 + Java 17
- **注册与配置中心**: Nacos
- **网关**: Spring Cloud Gateway
- **RPC**: Apache Dubbo 3.X
- **熔断限流**: Sentinel
- **分布式事务**: Seata
- **安全**: Sa-Token + JWT
- **数据库**: MySQL 8.4 + MyBatis Plus
- **缓存**: Redis 6 + Redisson
- **对象存储**: MinIO
- **消息队列**: RocketMQ
- **定时任务**: XXL-JOB
- **监控**: Prometheus + Grafana + SkyWalking + ELK

## 快速开始

### 环境要求

| 软件 | 版本要求 | 说明 |
|------|------|------|
| JDK | 17+ | Java 运行环境 |
| Maven | 3.6+ | 项目构建工具 |
| MySQL | 8.4+ | 主数据库 |
| Redis | 6.0+ | 缓存数据库 |
| Nacos | 2.0+ | 注册配置中心 |
| Node.js | 18+ | 前端开发环境 |

### 启动步骤

1. **克隆项目代码**

```bash
git clone https://gitee.com/earthy-zinc/dehaze-java-cloud-plus.git
```

2. **初始化数据库**

```bash
# 执行sql目录下的初始化脚本
mysql -u root -p < sql/mysql/*.sql
```

3. **启动 Nacos**

```bash
# 下载并启动Nacos
sh startup.sh -m standalone
```

4. **启动 Redis**

```bash
# 启动Redis服务
redis-server
```

5. **启动各服务模块**

```bash
# 启动网关服务
cd pei-gateway
mvn spring-boot:run

# 启动系统服务
cd pei-module-system
mvn spring-boot:run

# 启动基础设施服务
cd pei-module-infra
mvn spring-boot:run

# 启动其他业务模块（按需启动）
```

6. **访问系统**

- 管理后台: http://localhost:8080
- API 文档: http://localhost:8080/doc.html

## 部署

### Docker 部署

```bash
# 构建所有服务镜像
mvn clean install -Pdocker

# 启动所有服务
docker-compose up -d
```

### Kubernetes 部署

```bash
# 部署到K8s集群
kubectl apply -f k8s/
```
