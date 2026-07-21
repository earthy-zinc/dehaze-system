# 图像去雾系统（微服务版）

基于 RuoYi-Cloud-Plus 微服务架构构建的图像去雾系统。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Java微服务版架构文档.md)。

## 技术栈

- Spring Cloud Alibaba + Nacos + Spring Cloud Gateway
- Sa-Token + JWT 认证、MyBatis-Plus、Redisson
- MySQL 8.4 + Redis + MinIO + Elasticsearch
- Prometheus + Grafana 监控、SkyWalking 链路追踪、ELK 日志

## 快速开始

### 基础设施启动

必须先启动：MySQL、Redis、Nacos（导入 `script/config/` 下的配置）

可选：MinIO、Seata、Sentinel、SnailJob

### 应用服务启动顺序

必须启动：Gateway (pei-gateway) → Auth (pei-auth) → System (pei-modules/pei-system)

可选启动的应用服务：Resource 资源服务、Workflow 工作流服务、Gen 代码生成服务、Job 定时任务服务、Demo 演示服务

### 本地启动步骤

1. 执行 `script/sql/` 下的数据库脚本
2. 启动 Nacos，导入 `script/config/` 配置文件
3. 按顺序启动 Gateway、Auth、System 服务

### Docker 启动

```bash
mvn clean install -Pdocker
cd script/docker && docker-compose up -d
```

通过 Docker 方式可一键启动所有依赖服务和应用服务，包括：MySQL、Redis、Nacos、MinIO、Seata、Sentinel、Gateway、Auth、System 等应用服务，以及 Prometheus、Grafana 等监控相关服务。

## 部署

### 访问地址

| 服务 | 地址 |
|------|------|
| 管理后台 | http://localhost:8080 |
| API 文档 | http://localhost:8080/doc.html |
| Nacos 控制台 | http://localhost:8848 |
| Sentinel 控制台 | http://localhost:8718 |
| Seata 控制台 | http://localhost:7091 |
| MinIO 控制台 | http://localhost:9001 |
