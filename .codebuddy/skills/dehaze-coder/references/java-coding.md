# Java 编程规范

适用于：`dehaze-java*` 系列后端项目（Spring Boot / MyBatis-Plus）

> 项目架构与基础设施详见 `dehaze-doc/docs/05-子项目实现/Java后端基础设施文档.md`

## 优先级

1. 分层与职责边界明确：Controller / Service / Mapper（Repository）
2. 依赖注入可测试：构造注入优先，外部依赖必须可替换（接口/网关封装）
3. 输入校验与安全：Bean Validation + 权限边界 + 日志可观测

## 外部服务调用必须可替换

- MinIO/OSS/HTTP/XXL-Job：封装为 `xxxClient/xxxGateway` 接口，并通过构造注入
- 禁止：业务代码里 `new SDK client`；禁止在业务逻辑里读环境变量拼配置

## 多存储一致性

- 明确每个存储的职责：MySQL 主数据、Redis 缓存、Mongo 文档/日志等
- 禁止在一个方法中随意混用多个存储导致事务/一致性不可控
- 如必须混用：写清楚补偿策略或最终一致性说明

## 校验与异常处理

- 入参校验：Bean Validation（`@Valid`、自定义校验器等）
- 统一异常处理：`@ControllerAdvice` + `@ExceptionHandler`（返回结构一致）

## 代码规模限制

- 单文件不超过 800 行
- 单函数不超过 80 行
- 单元测试文件不超过 1600 行；单测函数不超过 160 行
