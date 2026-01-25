---
# 注意不要修改本文头文件，如修改，CodeBuddy（内网版）将按照默认逻辑设置
type: always
---
# Java 编程规范

## 优先级

1. 分层与职责边界明确：Controller / Service / Mapper（Repository）
2. 依赖注入可测试：构造注入优先，外部依赖必须可替换（接口/网关封装）
3. 输入校验与安全：Bean Validation + 权限边界 + 日志可观测

## 架构与分层

- RESTful：HTTP 方法/状态码语义正确
- Controller：只做入参校验、DTO/VO 转换、权限/调用编排；不写业务规则
- Service：承载业务逻辑与事务边界；复杂流程写清楚一致性策略
- Mapper（MyBatis-Plus）：只做数据访问，不承载业务逻辑
    - 复杂查询推荐 XML，且保持统一风格
    - 分页优先使用 MyBatis-Plus 分页规范，不在 Controller 手写分页计算

## 配置与属性

- 使用 `application.yml/properties` + Profiles 区分环境
- 优先用 `@ConfigurationProperties` 管理配置（类型安全）

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

## 安全

- Spring Security：认证/授权边界清晰
- 密码：使用 BCrypt 等安全编码
- 必要时配置 CORS

## 日志与监控

- 日志：SLF4J + Logback，等级合理（ERROR/WARN/INFO/DEBUG）
- 监控：Actuator 指标与健康检查

## 代码规模限制

- 单文件不超过 800 行
- 单函数不超过 80 行
- 单元测试文件不超过 1600 行；单测函数不超过 160 行

## API 文档

- 使用 Knife4j / springdoc
- DTO/VO 字段要有清晰注释与校验注解（`@Schema` / `@NotNull` 等）
- 安全接口标注鉴权方式（如 Bearer token）