# Java 编程规范

适用于：`dehaze-java*` 系列后端项目（Spring Boot / MyBatis-Plus）

> 项目架构与基础设施详见 `dehaze-doc/docs/04-项目实现/后端/01-Java架构文档.md`

## 外部服务调用必须可替换

- MinIO/OSS/HTTP/XXL-Job：封装为 `xxxClient/xxxGateway` 接口，并通过构造注入
- 禁止：业务代码里 `new SDK client`；禁止在业务逻辑里读环境变量拼配置

## 多存储一致性

- 明确每个存储的职责：MySQL 主数据、Redis 缓存、Mongo 文档/日志等
- 禁止在一个方法中随意混用多个存储导致事务/一致性不可控
- 如必须混用：写清楚补偿策略或最终一致性说明
