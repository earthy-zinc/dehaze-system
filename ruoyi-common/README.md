
### `ruoyi-common` 包分析

`ruoyi-common` 是一个典型的 **通用工具库模块**，其设计目的是为整个项目提供可复用的公共组件、基础类库和通用功能。它被划分为多个子包，每个子包专注于某一特定功能领域。

---

## 一、为什么单独构建 `ruoyi-common` 及其子包？

### 1. **代码复用性**
- 将常用功能封装为独立模块，避免重复开发。
- 多个业务模块（如 `ruoyi-system`, `ruoyi-resource`）可以统一引用这些公共组件。

### 2. **模块解耦**
- 每个子包职责单一，降低模块间耦合度。
- 如 `ruoyi-common-redis` 只负责 Redis 相关操作，`ruoyi-common-sms` 只处理短信发送逻辑。

### 3. **版本管理与依赖控制**
- 使用 `ruoyi-common-bom` 统一管理所有子包的版本。
- 避免不同模块引入不一致版本导致的问题。

### 4. **便于维护与扩展**
- 当需要升级某个通用功能时，只需修改对应的子包，不影响其他模块。
- 新增功能只需新增子包，不影响现有结构。

---

## 二、各主要子包作用详解

| 子包名称 | 作用 | 设计原因 |
|----------|------|-----------|
| **ruoyi-common-bom** | 提供所有 [ruoyi-common-*](file://E:\ProgramProject\RuoYi-Cloud-Plus\ruoyi-common\ruoyi-common-bom\pom.xml) 子包的依赖版本管理 | 统一版本，避免冲突 |
| **ruoyi-common-core** | 核心工具类（如字符串处理、日期格式化、常量定义等） | 基础支持，几乎所有模块都会依赖 |
| **ruoyi-common-dict** | 字典数据管理 | 统一系统中使用的静态数据（如性别、状态码） |
| **ruoyi-common-dubbo** | Dubbo 服务调用支持 | 微服务间通信标准化 |
| **ruoyi-common-elasticsearch** | Elasticsearch 操作封装 | 支持全文检索功能 |
| **ruoyi-common-encrypt** | 加密解密工具类（如 AES、MD5、SHA） | 数据安全处理 |
| **ruoyi-common-excel** | Excel 导入导出支持（基于 Apache POI） | 数据导入导出标准化 |
| **ruoyi-common-idempotent** | 幂等性校验工具 | 防止重复提交 |
| **ruoyi-common-job** | 定时任务调度支持 | 统一任务调度机制 |
| **ruoyi-common-json** | JSON 序列化/反序列化工具（如 Jackson、FastJSON） | 数据交换标准格式 |
| **ruoyi-common-loadbalancer** | 负载均衡策略配置 | 微服务调用时负载均衡支持 |
| **ruoyi-common-log** | 日志记录工具封装（如 AOP 记录请求日志） | 统一日志输出格式 |
| **ruoyi-common-mail** | 邮件发送支持 | 系统通知、告警等功能 |
| **ruoyi-common-mybatis** | MyBatis 扩展支持（如自动填充、乐观锁） | ORM 操作增强 |
| **ruoyi-common-nacos** | Nacos 配置中心集成 | 动态配置管理 |
| **ruoyi-common-oss** | 对象存储服务封装（如阿里云 OSS、MinIO） | 文件上传下载标准化 |
| **ruoyi-common-prometheus** | Prometheus 监控指标暴露 | 系统运行状态监控 |
| **ruoyi-common-ratelimiter** | 限流工具（如 Guava RateLimiter） | 控制接口访问频率 |
| **ruoyi-common-redis** | Redis 缓存操作封装 | 提升系统响应速度 |
| **ruoyi-common-satoken** | Sa-Token 权限框架集成 | 统一认证授权机制 |
| **ruoyi-common-seata** | Seata 分布式事务集成 | 保证跨服务事务一致性 |
| **ruoyi-common-security** | Spring Security 集成 | 权限控制基础支持 |
| **ruoyi-common-sensitive** | 敏感词过滤工具 | 内容安全控制 |
| **ruoyi-common-sentinel** | Alibaba Sentinel 流控组件集成 | 服务熔断降级支持 |
| **ruoyi-common-skylog** | SkyWalking 日志集成 | 分布式链路追踪支持 |
| **ruoyi-common-sms** | 短信服务封装（如阿里云 SMS、腾讯云 SMS） | 通知、验证码发送 |
| **ruoyi-common-social** | 社交登录支持（如微信、QQ、GitHub 登录） | 第三方登录集成 |
| **ruoyi-common-sse** | Server-Sent Events 支持 | 实时消息推送 |
| **ruoyi-common-tenant** | 多租户支持 | SaaS 架构下隔离不同租户数据 |
| **ruoyi-common-translation** | 国际化多语言支持 | 多语言切换 |
| **ruoyi-common-web** | Web 层通用支持（如 BaseController、ResponseEntity） | 统一返回格式、异常处理 |
| **ruoyi-common-websocket** | WebSocket 支持 | 实时通信 |

---

## 三、最佳实践建议

| 类型 | 最佳实践 |
|------|----------|
| **模块划分** | 按功能细粒度拆分，保持每个子包职责单一。 |
| **版本管理** | 使用 BOM 统一管理依赖版本，确保一致性。 |
| **日志记录** | 所有公共组件应支持日志输出，便于调试。 |
| **异常处理** | 自定义统一异常类型，避免底层错误暴露。 |
| **缓存机制** | 对高频读取操作增加本地或分布式缓存。 |
| **幂等性控制** | 对写操作接口增加 Token 或唯一 ID 校验。 |
| **测试覆盖** | 所有核心工具类应有单元测试，确保稳定性。 |

---

## ✅ 总结

`ruoyi-common` 及其子包的设计体现了良好的模块化思想和面向接口编程理念。通过将通用功能抽象为独立模块，不仅提升了系统的可维护性和可扩展性，也为后续新功能的接入提供了统一的标准和规范。

这种设计非常适合企业级微服务架构，能够有效支撑复杂系统的长期演进与迭代。