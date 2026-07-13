# 后端基础设施设计（dehaze-java）

## 1. 文档概述

### 1.1 文档目的

本文档描述 `dehaze-java` 后端项目的基础设施层设计，包括项目分层架构、应用生命周期、配置管理、数据访问层、缓存体系、消息队列、定时任务、安全过滤器链、日志系统和可观测性等基础能力。

本文档**不涉及**具体业务模块的实现逻辑，业务模块详见 [模块设计](../03-模块设计/) 各子目录。

### 1.2 适用范围

面向参与 `dehaze-java` 后端开发的工程师，提供技术基座的全局视图和设计决策依据。

### 1.3 相关文档

| 文档 | 说明 |
|------|------|
| [01-总体架构设计](./01-总体架构设计.md) | 系统全局分层、数据流与安全策略 |
| [03-数据库设计](./03-数据库设计.md) | 表结构、索引、ER 关系图 |
| [04-API 规范](./04-API规范.md) | 全局 API 规范、认证方式、错误码 |
| [07-后端基础设施设计(Go)](./07-后端基础设施设计(Go).md) | Go 端对等基础设施设计 |

---

## 2. 项目目录结构

```
dehaze-java/
├── pom.xml                             # Maven 依赖管理
├── src/
│   ├── main/
│   │   ├── java/com/pei/dehaze/
│   │   │   ├── SystemApplication.java  # SpringBoot 启动入口
│   │   │   ├── common/                 # 公共基础模块
│   │   │   │   ├── base/               # 基类（BaseEntity/BasePageQuery/IBaseEnum）
│   │   │   │   ├── constant/           # 常量定义（JWT/Security/Task）
│   │   │   │   ├── enums/              # 业务枚举（状态/类型/权限范围）
│   │   │   │   ├── exception/          # 异常体系（BusinessException + 全局处理器）
│   │   │   │   ├── model/              # 公共模型（Option）
│   │   │   │   ├── result/             # 统一响应（Result/ResultCode/PageResult）
│   │   │   │   ├── util/               # 工具类（XSS/路径安全/文件/日期）
│   │   │   │   └── validator/          # 自定义校验注解
│   │   │   ├── config/                 # 配置类
│   │   │   │   ├── property/           # 配置属性类（@ConfigurationProperties）
│   │   │   │   ├── SecurityConfig.java # Spring Security 过滤器链
│   │   │   │   ├── MybatisConfig.java  # MyBatis-Plus 分页/数据权限/自动填充
│   │   │   │   ├── RedisConfig.java    # Redis 序列化
│   │   │   │   ├── RedisCacheConfig.java # Spring Cache 集成
│   │   │   │   ├── CorsConfig.java     # 跨域配置
│   │   │   │   ├── AsyncConfig.java    # 异步线程池
│   │   │   │   ├── WebMvcConfig.java   # MVC 序列化/校验
│   │   │   │   ├── WebSocketConfig.java # WebSocket STOMP
│   │   │   │   ├── SwaggerConfig.java  # Knife4j API 文档
│   │   │   │   ├── XxlJobConfig.java   # XXL-Job 定时任务
│   │   │   │   └── CaptchaConfig.java  # 验证码
│   │   │   ├── filter/                 # Servlet 过滤器
│   │   │   │   ├── JwtValidationFilter.java
│   │   │   │   ├── CaptchaValidationFilter.java
│   │   │   │   └── RequestLogFilter.java
│   │   │   ├── security/              # 安全组件
│   │   │   │   ├── exception/          # 认证/授权异常处理器
│   │   │   │   ├── model/              # SysUserDetails
│   │   │   │   ├── service/            # UserDetailsService/PermissionService
│   │   │   │   └── util/               # JwtUtils/SecurityUtils
│   │   │   ├── plugin/                 # 插件化扩展组件
│   │   │   │   ├── mybatis/            # MyBatis 插件（数据权限/自动填充）
│   │   │   │   ├── dupsubmit/          # 防重复提交（AOP + Redisson）
│   │   │   │   ├── ratelimit/          # 接口限流（AOP + Redisson）
│   │   │   │   ├── captcha/            # 验证码配置属性
│   │   │   │   ├── easyexcel/          # Excel 导入监听器
│   │   │   │   └── xxljob/             # XXL-Job 任务处理器
│   │   │   ├── controller/             # Controller 层（请求处理）
│   │   │   ├── service/                # Service 层（业务逻辑）
│   │   │   │   ├── impl/               # Service 实现
│   │   │   │   │   └── file/           # 文件存储策略实现
│   │   │   │   └── strategy/           # 任务策略模式
│   │   │   ├── mapper/                 # Mapper 层（数据访问）
│   │   │   ├── converter/              # MapStruct 对象转换器
│   │   │   ├── model/                  # 数据模型
│   │   │   │   ├── entity/             # 数据库实体
│   │   │   │   ├── bo/                 # 业务对象
│   │   │   │   ├── dto/                # 数据传输对象
│   │   │   │   ├── vo/                 # 视图对象（API 响应）
│   │   │   │   ├── form/               # 表单对象（API 入参）
│   │   │   │   ├── query/              # 查询条件对象
│   │   │   │   └── event/              # 领域事件
│   │   │   ├── job/                    # 定时任务
│   │   │   └── listener/              # 事件监听器
│   │   └── resources/
│   │       ├── application.yml         # 主配置（profile 切换）
│   │       ├── application-dev.yml     # 开发环境
│   │       ├── application-prod.yml    # 生产环境
│   │       ├── logback-spring.xml      # 日志配置
│   │       ├── mapper/                 # MyBatis XML 映射文件
│   │       └── excel-templates/        # Excel 导入模板
│   └── test/
│       ├── java/com/pei/dehaze/
│       │   ├── base/                   # 测试基类
│       │   ├── controller/             # Controller 测试
│       │   ├── service/                # Service 测试
│       │   └── generator/              # 代码生成器
│       └── resources/
│           ├── application-test.yml    # 测试配置
│           ├── application-test-h2.yml # H2 内存数据库测试
│           ├── application-test-tc.yml # TestContainers 测试
│           ├── db/                     # 测试 SQL（init.sql + h2/，schema.sql 和 data.sql 由 Maven 从 config/sql/ 自动复制）
│           └── templates/              # 代码生成模板
```

---

## 3. 分层架构设计

### 3.1 架构分层

项目采用**经典分层架构 + Spring IoC 容器自动装配**的设计。

```mermaid
flowchart TB
    subgraph External["外部请求"]
        Client["HTTP Client"]
    end

    subgraph FilterChain["Servlet Filter 链"]
        direction LR
        CORS["跨域 CorsFilter"]
        Captcha["验证码 CaptchaFilter"]
        JWT["JWT JwtValidationFilter"]
        SecurityChain["Spring Security FilterChain"]
    end

    subgraph Controller["Controller 层 (controller/)"]
        direction LR
        C1["参数绑定"]
        C2["参数校验"]
        C3["响应封装"]
    end

    subgraph Service["Service 层 (service/)"]
        direction LR
        S1["业务编排"]
        S2["事务管理"]
        S3["缓存策略"]
    end

    subgraph Mapper["Mapper 层 (mapper/)"]
        direction LR
        M1["CRUD 封装"]
        M2["XML SQL"]
        M3["数据权限拦截"]
    end

    subgraph Infrastructure["基础设施 (config/ + plugin/)"]
        direction LR
        DB[("MySQL")]
        Cache[("Redis")]
        MQ[("RabbitMQ")]
        Log["Logback"]
    end

    Client --> FilterChain --> Controller --> Service --> Mapper --> Infrastructure
```

### 3.2 层级职责

| 层级 | 包路径 | 职责 | 依赖方向 |
|------|--------|------|----------|
| **Filter 链** | `filter/` + Spring Security | 请求拦截、JWT 校验、验证码、跨域 | ← 外部请求 |
| **Controller 层** | `controller/` | 参数绑定与校验、调用 Service、统一响应 | → Service |
| **Service 层** | `service/` | 业务逻辑编排、事务边界、缓存交互 | → Mapper + plugin |
| **Mapper 层** | `mapper/` | 数据库 CRUD、SQL 构建、数据权限 | → MyBatis-Plus |
| **基础设施层** | `config/` + `plugin/` + `common/` | 配置、缓存、安全、限流等基础能力 | 被所有层依赖 |

### 3.3 依赖注入策略

项目基于 **Spring IoC 容器**实现自动依赖装配：

- 使用 `@RequiredArgsConstructor` (Lombok) 生成构造函数注入，优先于字段注入
- 配置类使用 `@Configuration` + `@Bean` 显式声明基础组件
- 条件装配使用 `@ConditionalOnProperty` 控制组件按需加载（如 XXL-Job、Redis Cache）
- 属性绑定使用 `@ConfigurationProperties` + `@ConfigurationPropertiesScan`

```mermaid
flowchart LR
    Boot["SpringBoot 自动配置"] --> Config["@Configuration 显式声明"]
    Config --> Bean["@Bean 组件注册"]
    Bean --> DI["@RequiredArgsConstructor 构造注入"]
    DI --> Runtime["运行时依赖解析"]
```

**设计决策**：选择 Spring 原生 IoC 而非其他 DI 框架（如 Guice），因为 SpringBoot 生态天然支持，无需额外引入。通过 `@ConditionalOnProperty` 实现按环境装配，避免测试环境加载生产专用组件。

### 3.4 数据模型分层

```mermaid
flowchart LR
    Request["HTTP Request"] --> Form["Form (表单对象)<br/>入参绑定 + 校验"]
    Form --> Entity["Entity (实体)<br/>数据库表映射"]
    Entity --> VO["VO (视图对象)<br/>API 响应"]

    Query["Query (查询对象)<br/>分页 + 过滤"] --> Mapper
    BO["BO (业务对象)<br/>内部业务传递"] -.-> Service
    DTO["DTO (传输对象)<br/>服务间传递"] -.-> Service
    Event["Event (领域事件)<br/>事件驱动"] -.-> Listener
```

| 模型类型 | 包路径 | 职责 | 示例 |
|----------|--------|------|------|
| **Entity** | `model/entity/` | 数据库表映射，MyBatis-Plus 注解 | `SysUser` |
| **Form** | `model/form/` | 请求入参绑定、校验注解 | `UserForm` |
| **VO** | `model/vo/` | API 响应输出、Swagger 注解 | `UserPageVO` |
| **Query** | `model/query/` | 分页查询条件 | `UserPageQuery` |
| **BO** | `model/bo/` | 业务层内部传递 | `UserBO` |
| **DTO** | `model/dto/` | 服务间数据传输 | `LoginResult` |
| **Event** | `model/event/` | 领域事件载荷 | `ItemFileCreatedEvent` |

**对象转换**：使用 MapStruct 编译期生成转换代码（`converter/` 包），避免运行时反射开销。

---

## 4. 应用生命周期管理

### 4.1 启动流程

```mermaid
sequenceDiagram
    participant Main as SystemApplication
    participant Boot as SpringBoot
    participant AutoConfig as 自动配置
    participant Bean as Bean 初始化
    participant Server as 内嵌 Tomcat

    Main->>Boot: SpringApplication.run()
    Boot->>Boot: 加载 application.yml
    Note right of Boot: Profile 切换<br/>环境变量展开
    Boot->>AutoConfig: @EnableAutoConfiguration
    Note right of AutoConfig: 数据源/Redis/Security<br/>MVC/Cache/Actuator
    AutoConfig->>Bean: @Configuration 扫描
    Note right of Bean: SecurityConfig<br/>MybatisConfig<br/>RedisConfig<br/>CorsConfig<br/>AsyncConfig<br/>XxlJobConfig(条件)
    Bean->>Bean: @ConfigurationPropertiesScan
    Note right of Bean: SecurityProperties<br/>CaptchaProperties
    Bean->>Bean: @ComponentScan
    Note right of Bean: Service/Mapper/Controller<br/>Filter/Plugin/Job
    Bean->>Server: 启动内嵌 Tomcat
    Server->>Server: 注册 Filter 链
    Note right of Server: CorsFilter(order=-101)<br/>CaptchaFilter<br/>JwtFilter<br/>SecurityFilterChain
```

### 4.2 优雅关闭

SpringBoot 内置 Graceful Shutdown 支持：

```
收到 SIGINT/SIGTERM
    → Tomcat 停止接收新连接
    → 等待 in-flight 请求完成（默认 30s 超时）
    → 销毁 Spring Bean（@PreDestroy）
    → 关闭数据源连接池（Druid）
    → 关闭 Redis 连接池（Lettuce）
    → 关闭线程池（ThreadPoolTaskExecutor）
```

### 4.3 多 Profile 配置

通过 `spring.profiles.active` 控制环境切换：

```yaml
# application.yml
spring:
  profiles:
    active: dev  # 可选: dev / prod / test
```

---

## 5. 配置管理

### 5.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 配置加载 | Spring Boot YAML | 多 Profile、环境变量覆盖 |
| 属性绑定 | `@ConfigurationProperties` | 类型安全的配置绑定 |
| 环境变量 | `${ENV_VAR}` 占位符 | 敏感信息外部化 |
| 参数校验 | Hibernate Validator | JSR-380 Bean Validation |

### 5.2 配置结构

```yaml
# application-dev.yml 配置结构概览
server:          # 服务端口
spring:
  datasource:    # 数据源配置（Druid 连接池）
  data:
    redis:       # Redis 配置（Lettuce 连接池）
    mongodb:     # MongoDB 配置
  cache:         # Spring Cache 配置（Redis 后端）
  jackson:       # JSON 序列化配置
  servlet:       # 文件上传限制
security:        # 安全配置（JWT 密钥/TTL、忽略路径）
management:      # Actuator 监控配置
file:            # 文件存储配置（MinIO/Local）
springdoc:       # Swagger API 文档配置
knife4j:         # Knife4j 增强配置
xxl:             # XXL-Job 定时任务配置
captcha:         # 验证码配置
mybatis-plus:    # ORM 配置
```

### 5.3 多环境支持

| 环境 | Profile | 配置文件 | 特性差异 |
|------|---------|----------|----------|
| **开发** | dev | `application-dev.yml` | SQL 日志输出、Swagger 启用、DevTools 热重载 |
| **测试** | test | `application-test.yml` | H2 内存数据库 / TestContainers、缓存禁用 |
| **生产** | prod | `application-prod.yml` | Swagger 禁用、连接池优化、日志输出到文件 |

### 5.4 敏感信息管理

敏感配置通过环境变量注入，YAML 中使用 `${ENV_VAR}` 占位符：

```yaml
spring:
  datasource:
    password: ${DEHAZE_PASSWORD}
  data:
    redis:
      password: ${DEHAZE_PASSWORD}
    mongodb:
      uri: mongodb://root:${DEHAZE_PASSWORD}@host:27017/dehaze?authSource=admin
```

### 5.5 条件化装配

通过 `@ConditionalOnProperty` 按配置开关控制组件加载：

| 组件 | 配置项 | 说明 |
|------|--------|------|
| XXL-Job | `xxl.job.enabled=true` | 定时任务执行器按需加载 |
| Redis Cache | `spring.cache.enabled=true` | Spring Cache 按需启用 |

---

## 6. 数据访问层

### 6.1 技术选型

| 组件 | 选型 | 版本 | 说明 |
|------|------|------|------|
| ORM | MyBatis-Plus | 3.5.5 | 通用 CRUD、分页、数据权限 |
| 连接池 | Druid | 1.2.16 | 监控、防 SQL 注入、连接管理 |
| 数据库 | MySQL | 9.5.0 (驱动) | 生产环境 |
| 测试数据库 | H2 / TestContainers(MySQL) | - | 单元测试 / 集成测试 |

### 6.2 连接池配置

```yaml
druid:
  initial-size: 5          # 初始连接数
  min-idle: 5              # 最小空闲连接
  max-active: 50           # 最大活跃连接数
  max-wait: 5000           # 获取连接最大等待时间(ms)
  time-between-eviction-runs-millis: 60000   # 空闲连接检查周期
  min-evictable-idle-time-millis: 300000     # 连接最小空闲时间
  validation-query: SELECT 1
  test-while-idle: true
  filters: stat,wall       # 统计监控 + SQL 防火墙
```

### 6.3 MyBatis-Plus 插件链

```mermaid
flowchart LR
    SQL["SQL 执行"] --> DP["DataPermissionInterceptor<br/>数据权限拦截"]
    DP --> Page["PaginationInnerInterceptor<br/>分页插件"]
    Page --> DB[("MySQL")]
```

| 插件 | 类名 | 功能 |
|------|------|------|
| 数据权限 | `DataPermissionInterceptor` | 基于 `@DataPermission` 注解自动追加 SQL 条件 |
| 分页 | `PaginationInnerInterceptor` | 自动分页，支持 MySQL 方言 |

### 6.4 自动填充（MetaObjectHandler）

| 触发时机 | 填充字段 | 值来源 |
|----------|----------|--------|
| INSERT | `createTime` / `updateTime` | `LocalDateTime.now()` |
| INSERT | `createBy` / `updateBy` | `SecurityUtils.getUserId()` |
| UPDATE | `updateTime` | `LocalDateTime.now()` |
| UPDATE | `updateBy` | `SecurityUtils.getUserId()` |

### 6.5 数据权限（DataScope）

基于 MyBatis-Plus `DataPermissionHandler` 实现行级数据权限控制，通过 `@DataPermission` 注解标记需要权限控制的 Mapper 方法：

| 权限范围 | 枚举值 | SQL 效果 |
|----------|--------|----------|
| 全部数据 | `ALL` | 无附加条件 |
| 本部门数据 | `DEPT` | `WHERE dept_id = ?` |
| 本部门及下级 | `DEPT_AND_SUB` | `WHERE dept_id IN (SELECT id FROM sys_dept WHERE ...)` |
| 仅本人数据 | `SELF` | `WHERE create_by = ?` |

### 6.6 逻辑删除

全局配置逻辑删除：

```yaml
mybatis-plus:
  global-config:
    db-config:
      logic-delete-field: deleted
      logic-delete-value: 1
      logic-not-delete-value: 0
```

---

## 7. 缓存体系

### 7.1 架构设计

```mermaid
flowchart TB
    subgraph CacheArch["缓存体系"]
        direction TB

        subgraph SpringCache["Spring Cache (注解式)"]
            Annotation["@Cacheable / @CacheEvict<br/>声明式缓存"]
        end

        subgraph RedisTemplate["RedisTemplate (编程式)"]
            KV["Key-Value 操作"]
            Hash["Hash 操作"]
        end

        subgraph Redisson["Redisson"]
            Lock["分布式锁"]
            RateLimit["限流器"]
        end
    end

    subgraph Backend["Redis 6.0+"]
        Redis[("Redis<br/>Lettuce 连接池")]
    end

    SpringCache --> Redis
    RedisTemplate --> Redis
    Redisson --> Redis
```

### 7.2 Spring Cache 配置

| 配置项 | 值 | 说明 |
|--------|------|------|
| 后端类型 | Redis | `spring.cache.type=redis` |
| 默认 TTL | 3600s | `spring.cache.redis.time-to-live` |
| 空值缓存 | 启用 | 防止缓存穿透 |
| Key 序列化 | StringRedisSerializer | 可读性 |
| Value 序列化 | GenericJackson2JsonRedisSerializer | JSON 格式，支持 JSR-310 时间类型 |
| Key 前缀 | `cacheName:` | 覆盖默认双冒号分隔符 |

### 7.3 Redis 序列化

`RedisTemplate<String, Object>` 自定义序列化，避免 JDK 默认序列化的可读性问题：

```
Key:   StringRedisSerializer (UTF-8 字符串)
Value: GenericJackson2JsonRedisSerializer (JSON，含类型信息)
Hash Key:   StringRedisSerializer
Hash Value: GenericJackson2JsonRedisSerializer
```

### 7.4 分布式锁（Redisson）

| 使用场景 | 实现方式 | 说明 |
|----------|----------|------|
| 防重复提交 | `RLock` + `tryLock` | 基于 Token JTI + 请求路径生成锁 Key |
| 接口限流 | `RRateLimiter` | 基于 Redisson 令牌桶算法 |

### 7.5 缓存演进规划

与 Go 端保持一致的演进方向：

```
当前: Spring Cache → Redis (单级缓存)
规划: L1 本地缓存 (Caffeine) + L2 Redis (多级缓存)
      + 布隆过滤器 (穿透防护)
      + SingleFlight (击穿防护)
      + Pub/Sub (多实例 L1 失效广播)
```

---

## 8. 消息队列

### 8.1 技术选型

与 Go 端保持一致的中间件选型：

| 消息中间件 | 用途 | 当前状态 |
|------------|------|----------|
| **RabbitMQ** | 异步任务分发（导出、批量操作等） | 📋 规划中（当前使用 @Async 线程池） |
| **Kafka** | 日志收集与流处理 | 📋 规划中 |

### 8.2 当前方案：@Async 异步线程池

当前使用 Spring `@Async` + 自定义线程池处理异步任务：

```java
@Bean("datasetTaskExecutor")
public Executor datasetTaskExecutor() {
    ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
    executor.setCorePoolSize(2);
    executor.setMaxPoolSize(4);
    executor.setQueueCapacity(10);
    executor.setThreadNamePrefix("dataset-async-");
    executor.setRejectedExecutionHandler(new CallerRunsPolicy());
    return executor;
}
```

配合**策略模式** (`TaskStrategy`) 实现任务类型路由：

```mermaid
flowchart LR
    subgraph TaskSystem["异步任务系统"]
        Controller["TaskController"] --> TaskService
        TaskService --> TaskExecutor["TaskExecutorImpl<br/>@Async"]
        TaskExecutor --> Factory["TaskStrategyFactory"]
        Factory --> S1["DatasetExportStrategy"]
        Factory --> S2["BatchDownloadStrategy"]
        Factory --> S3["CustomExportStrategy"]
        Factory --> S4["ItemDownloadStrategy"]
    end

    subgraph Progress["进度回调"]
        WebSocket["WebSocket STOMP"]
    end

    TaskExecutor -.->|"ProgressCallback"| WebSocket
```

### 8.3 RabbitMQ 集成规划

与 Go 端对齐，采用相同的交换机和队列设计：

```mermaid
flowchart LR
    subgraph Producer["生产者 (dehaze-java)"]
        TaskService2["TaskService"]
        Publisher["RabbitTemplate"]
    end

    subgraph RabbitMQ["RabbitMQ Broker"]
        Exchange["dehaze.tasks<br/>(direct exchange)"]
        Q1["task.export<br/>导出任务队列"]
        Q2["task.download<br/>批量下载队列"]
        Q3["task.thumbnail<br/>缩略图生成队列"]
        DLX["Dead Letter Exchange<br/>死信交换机"]
    end

    subgraph Consumer["消费者 (dehaze-java)"]
        Worker1["ExportWorker"]
        Worker2["DownloadWorker"]
        Worker3["ThumbnailWorker"]
    end

    TaskService2 --> Publisher
    Publisher --> Exchange
    Exchange -->|"routing_key"| Q1
    Exchange -->|"routing_key"| Q2
    Exchange -->|"routing_key"| Q3
    Q1 --> Worker1
    Q2 --> Worker2
    Q3 --> Worker3
    Q1 -.->|"nack/超时"| DLX
    Q2 -.->|"nack/超时"| DLX
    Q3 -.->|"nack/超时"| DLX
```

**迁移计划**：

| 阶段 | 内容 |
|------|------|
| **Phase 1** | 引入 `spring-boot-starter-amqp`，配置 RabbitMQ 连接 |
| **Phase 2** | 将现有 @Async 任务迁移为 RabbitMQ 消费者，复用策略模式 |
| **Phase 3** | 与 Go 端共享队列，实现跨语言任务协作 |

### 8.4 Kafka 规划（日志管道）

与 Go 端保持一致：

```mermaid
flowchart LR
    subgraph Sources["日志源"]
        AppLog["应用日志"]
        AuditLog["审计日志"]
        AccessLog["访问日志"]
    end

    subgraph Kafka["Kafka Cluster"]
        Topic["dehaze.logs"]
    end

    subgraph Sink["日志存储"]
        ES["Elasticsearch<br/>(检索)"]
        S3["对象存储<br/>(归档)"]
    end

    Sources --> Topic --> Sink
```

---

## 9. 定时任务

### 9.1 当前方案：Spring @Scheduled + XXL-Job（条件装配）

| 调度方式 | 适用场景 | 当前状态 |
|----------|----------|----------|
| `@Scheduled` | 轻量级、单实例定时任务 | ✅ 已启用 |
| XXL-Job | 分布式调度、Web 管理 | ✅ 已集成（条件装配） |

### 9.2 内置定时任务

| 任务 | CRON | 功能 |
|------|------|------|
| `cleanupExpiredTasks` | `0 0 2 * * ?` | 每天凌晨 2 点清理 7 天前已完成任务、30 天前所有任务 |
| `cleanupStuckTasks` | `0 0 * * * ?` | 每小时清理超过 24 小时的异常状态任务 |

### 9.3 XXL-Job 集成

通过 `@ConditionalOnProperty(name = "xxl.job.enabled")` 按需装配：

```yaml
xxl:
  job:
    enabled: false                          # 开关
    admin:
      addresses: http://127.0.0.1:8080/xxl-job-admin
    accessToken: default_token
    executor:
      appname: xxl-job-executor-dehaze-java
      port: 9999
      logpath: ./logs/xxl-job
      logretentiondays: 30
```

```mermaid
flowchart LR
    subgraph XXLJob["XXL-Job Admin"]
        Scheduler["调度中心<br/>任务编排/日志/告警"]
    end

    subgraph JavaExecutor["dehaze-java Executor"]
        Handler1["TaskCleanupHandler"]
        Handler2["XxlJobSampleHandler"]
    end

    Scheduler -->|"HTTP 回调触发"| JavaExecutor
    JavaExecutor -->|"执行结果上报"| Scheduler
```

### 9.4 迁移计划

| 阶段 | 内容 |
|------|------|
| **Phase 1** | XXL-Job Admin 部署（与 Go 端共享） |
| **Phase 2** | 将 `@Scheduled` 任务迁移到 XXL-Job，保留代码但切换触发方式 |
| **Phase 3** | 新增统计/维护类任务直接注册到 XXL-Job |

---

## 10. 日志系统

### 10.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 日志框架 | SLF4J + Logback | SpringBoot 默认日志实现 |
| 日志切割 | TimeBasedRollingPolicy | 按天切割 + 按大小分片（10MB） |
| 日志保留 | maxHistory=15 | 保留 15 天 |

### 10.2 日志配置

```xml
<!-- logback-spring.xml 核心配置 -->
<configuration>
    <!-- 控制台输出 -->
    <appender name="CONSOLE">
        <filter><level>DEBUG</level></filter>
        <encoder><Pattern>${CONSOLE_LOG_PATTERN}</Pattern></encoder>
    </appender>

    <!-- 文件输出（按天切割 + 10MB 分片） -->
    <appender name="FILE">
        <file>./logs/${APP_NAME}/log.log</file>
        <rollingPolicy>
            <fileNamePattern>./logs/${APP_NAME}/%d{yyyy-MM-dd}.%i.log</fileNamePattern>
            <maxFileSize>10MB</maxFileSize>
            <maxHistory>15</maxHistory>
        </rollingPolicy>
        <filter><level>INFO</level></filter>
    </appender>

    <!-- 开发环境：控制台 + 文件 -->
    <springProfile name="dev">
        <root level="INFO">
            <appender-ref ref="CONSOLE"/>
            <appender-ref ref="FILE"/>
        </root>
    </springProfile>

    <!-- 生产环境：控制台 + 文件 -->
    <springProfile name="prod">
        <root level="INFO">
            <appender-ref ref="CONSOLE"/>
            <appender-ref ref="FILE"/>
        </root>
    </springProfile>
</configuration>
```

### 10.3 日志分层

| 层级 | 日志内容 | 示例 |
|------|----------|------|
| **Filter 层** | 请求 URI 日志 | `request uri: /api/v1/users` |
| **Controller 层** | Spring MVC 请求日志 | 由 RequestLogFilter 记录 |
| **Service 层** | 业务关键节点 | `用户登录成功 userId=123` |
| **Mapper 层** | SQL 日志（开发环境） | `==> Preparing: SELECT * FROM sys_user WHERE id = ?` |
| **异常处理** | 全局异常日志 | `biz exception: 用户不存在` |

### 10.4 日志演进规划

```
当前: SLF4J/Logback → 文件（按天切割）
规划: Logback → Kafka → Elasticsearch（检索）/ 对象存储（归档）
```

---

## 11. HTTP 服务与过滤器链

### 11.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | Spring Boot 3 + Spring MVC | 内嵌 Tomcat |
| API 文档 | Knife4j (OpenAPI 3) | 开发环境自动挂载 |
| WebSocket | Spring WebSocket + STOMP | 实时消息推送 |

### 11.2 过滤器链

请求经过的过滤器按注册顺序执行：

```mermaid
flowchart LR
    Req["请求"] --> CORS["跨域处理<br/>CorsFilter<br/>order=-101"]
    CORS --> Captcha["验证码校验<br/>CaptchaFilter"]
    Captcha --> JWT["JWT 校验<br/>JwtValidationFilter"]
    JWT --> Security["Spring Security<br/>FilterChain"]
    Security --> Permission["权限校验<br/>@PreAuthorize"]
    Permission --> Handler["业务处理<br/>Controller"]
```

### 11.3 过滤器清单

| 过滤器 | 文件 | 功能 | 作用范围 |
|--------|------|------|----------|
| CorsFilter | `CorsConfig.java` | 跨域资源共享，order=-101（先于 Security） | 全局 |
| CaptchaValidationFilter | `filter/CaptchaValidationFilter.java` | 登录验证码校验 | 登录接口 |
| JwtValidationFilter | `filter/JwtValidationFilter.java` | JWT Token 验证、SecurityContext 注入 | 受保护路由 |
| SecurityFilterChain | `SecurityConfig.java` | Spring Security 认证/授权链 | 全局 |
| RequestLogFilter | `filter/RequestLogFilter.java` | 请求 URI 日志 | 全局 |

### 11.4 Spring Security 过滤器链配置

```java
http
    .authorizeHttpRequests(registry ->
        registry
            .requestMatchers("/api/v1/auth/login").permitAll()
            .requestMatchers("/actuator/**").permitAll()
            .anyRequest().authenticated()
    )
    .sessionManagement(c -> c.sessionCreationPolicy(STATELESS))
    .csrf(AbstractHttpConfigurer::disable)
    .exceptionHandling(c ->
        c.authenticationEntryPoint(authenticationEntryPoint)    // 401
         .accessDeniedHandler(accessDeniedHandler)              // 403
    );
```

---

## 12. 安全基础设施

### 12.1 认证体系

| 组件 | 实现 | 说明 |
|------|------|------|
| JWT | Hutool JWT | AccessToken（`security.jwt.ttl` 秒有效期） |
| Token 黑名单 | Redis `BLACKLIST_TOKEN:` | 注销时将 JTI 加入黑名单 |
| 密码加密 | BCryptPasswordEncoder | Spring Security 标准实现 |
| 验证码 | Hutool Captcha | 支持圆圈/GIF/干扰线/扭曲多种类型 |
| UserDetails | SysUserDetailsService | 从数据库加载用户信息 |

### 12.2 权限体系

RBAC 权限模型通过 Spring Security `@PreAuthorize` + 自定义 `PermissionService` 实现：

```
用户 → 角色（多对多） → 权限标识（多对多）
权限格式: 模块:功能:操作（如 sys:user:add）
```

**权限校验流程**：

```mermaid
flowchart LR
    A["@PreAuthorize<br/>#40;@ss.hasPerm('sys:user:add')#41;"] --> B["PermissionService.hasPerm()"]
    B --> C{"超级管理员?"}
    C -->|是| D["直接放行"]
    C -->|否| E["从 Redis 获取角色权限"]
    E --> F{"包含所需权限?"}
    F -->|是| G["放行"]
    F -->|否| H["拒绝 403"]
```

权限缓存存储在 Redis Hash：`ROLE_PERMS_PREFIX → {roleCode: Set<String> perms}`

### 12.3 安全工具

| 工具 | 文件 | 功能 |
|------|------|------|
| XSS 过滤 | `util/XssUtils.java` | HTML 标签清洗 |
| 路径安全 | `util/PathSecurityUtil.java` | 路径穿越检测、安全路径拼接 |
| 安全上下文 | `security/util/SecurityUtils.java` | 获取当前用户 ID/部门/角色/数据权限 |
| JWT 工具 | `security/util/JwtUtils.java` | Token 解析、Authentication 构建 |

---

## 13. 插件化扩展组件

### 13.1 设计理念

通过 `plugin/` 包实现可插拔的横切关注点，每个插件独立封装为注解 + AOP 切面：

| 插件 | 包路径 | 注解 | 实现方式 |
|------|--------|------|----------|
| 防重复提交 | `plugin/dupsubmit/` | `@PreventDuplicateSubmit` | AOP + Redisson 分布式锁 |
| 接口限流 | `plugin/ratelimit/` | `@RateLimit` | AOP + Redisson 令牌桶 |
| 数据权限 | `plugin/mybatis/` | `@DataPermission` | MyBatis-Plus 拦截器 |
| 字段自动填充 | `plugin/mybatis/` | `@TableField(fill=...)` | MetaObjectHandler |

### 13.2 防重复提交

```mermaid
flowchart LR
    A["@PreventDuplicateSubmit<br/>expire=5s"] --> B["DuplicateSubmitAspect"]
    B --> C["生成锁 Key<br/>LOCK:RESUBMIT:{jti}:{method}-{uri}"]
    C --> D["Redisson tryLock"]
    D -->|获取成功| E["执行业务"]
    D -->|获取失败| F["抛出 REPEAT_SUBMIT_ERROR"]
    E --> G["释放锁"]
```

### 13.3 接口限流

```mermaid
flowchart LR
    A["@RateLimit<br/>type=IP, maxRequests=10, timeWindow=60"] --> B["RateLimitAspect"]
    B --> C["构建限流 Key<br/>rateLimit:{ip}:{class}#{method}"]
    C --> D["Redisson RRateLimiter"]
    D -->|获取令牌| E["执行业务"]
    D -->|令牌不足| F["抛出 RateLimitException"]
```

限流维度支持：IP / USER / GLOBAL。

---

## 14. 统一响应与错误处理

### 14.1 响应格式

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": { ... },
  "traceId": "xxx",
  "timestamp": 1234567890,
  "errors": []
}
```

### 14.2 错误码体系

错误码采用 **5 位字符串** 编码，与 Go 端保持一致：

| 前缀 | 类别 | 示例 |
|------|------|------|
| `00` | 成功 | `00000` - 一切 ok |
| `A0` | 用户端错误 | `A0001` - 用户端错误, `A0200` - 登录异常, `A0400` - 参数错误 |
| `B0` | 系统执行错误 | `B0001` - 系统执行出错, `B0210` - 并发限流 |
| `C0` | 第三方服务错误 | `C0001` - 调用第三方服务出错, `C0300` - 数据库服务出错 |

### 14.3 全局异常处理

通过 `@RestControllerAdvice` + `@ExceptionHandler` 统一拦截并格式化输出：

```mermaid
flowchart TB
    E1["BindException<br/>ConstraintViolationException<br/>MethodArgumentNotValidException"] -->|400| R1["ResultCode.PARAM_ERROR"]
    E2["BusinessException"] -->|400| R2["自定义 ResultCode"]
    E3["RateLimitException"] -->|429| R3["ResultCode.RATE_LIMIT"]
    E4["NoHandlerFoundException"] -->|404| R4["ResultCode.RESOURCE_NOT_FOUND"]
    E5["AccessDeniedException<br/>AuthenticationException"] -->|"继续抛出"| R5["Security 异常处理器"]
    E6["Exception (兜底)"] -->|400| R6["错误消息"]
```

**异常分类处理**：

| 异常类型 | HTTP 状态码 | 处理方式 |
|----------|-------------|----------|
| 参数校验异常 | 400 | 收集所有校验错误信息，拼接返回 |
| 业务异常 | 400 | 返回 BusinessException 中的 ResultCode |
| 限流异常 | 200 | 返回 RATE_LIMIT 错误码 |
| 资源不存在 | 404 | 返回 RESOURCE_NOT_FOUND |
| SQL 异常 | 403 | 检测是否为权限不足 |
| Security 异常 | - | 继续抛出交由 Spring Security 处理 |
| 未知异常 | 400 | 兜底处理，记录日志 |

---

## 15. 参数校验

### 15.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 校验框架 | Hibernate Validator (JSR-380) | Bean Validation 标准实现 |
| FailFast | 启用 | 遇到第一个校验失败立即返回 |
| 自定义校验 | `@FileExists` / `@DirExists` | 文件/目录存在性校验 |

### 15.2 校验策略

- Form 对象使用 `@NotBlank` / `@NotNull` / `@Size` / `@Pattern` 等注解声明校验规则
- Controller 使用 `@Valid` / `@Validated` 触发校验
- FailFast 模式：第一个校验失败立即返回，避免收集过多错误信息

### 15.3 自定义校验注解

| 注解 | 校验器 | 功能 |
|------|--------|------|
| `@FileExists` | `FileExistValidator` | 校验文件路径是否存在 |
| `@DirExists` | `DirExistValidator` | 校验目录路径是否存在 |

---

## 16. WebSocket 实时通信

### 16.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 协议 | STOMP over WebSocket | 消息语义化 |
| 兼容 | SockJS | 浏览器不支持 WebSocket 时降级 |
| 认证 | JWT Token 解析 | CONNECT 帧中提取用户信息 |

### 16.2 端点配置

| 端点 | 协议 | 说明 |
|------|------|------|
| `/ws` | WebSocket + SockJS | 浏览器客户端 |
| `/ws-app` | 原生 WebSocket | uni-app 客户端 |

### 16.3 消息代理

```
客户端发送前缀: /app
订阅前缀: /topic (广播) / /queue (点对点)
用户前缀: /user
```

---

## 17. 可观测性

### 17.1 指标采集（Metrics）

| 指标类型 | 实现方式 | 端点 |
|----------|----------|------|
| JVM 指标 | Micrometer + Prometheus | `/actuator/prometheus` |
| HTTP 请求指标 | Spring Boot Actuator | `/actuator/prometheus` |
| 自定义业务指标 | Micrometer API | `/actuator/prometheus` |

### 17.2 健康检查

| 端点 | 功能 |
|------|------|
| `GET /actuator/health` | 应用存活探针（含各组件健康状态） |
| `GET /actuator/info` | 应用信息 |
| `GET /actuator/metrics` | 指标列表 |

Actuator 配置：

```yaml
management:
  endpoints:
    web:
      exposure:
        include: "*"
  endpoint:
    health:
      show-details: ALWAYS
  metrics:
    enable:
      prometheus: true
```

### 17.3 API 文档

- 开发环境自动挂载 Knife4j UI：`GET /doc.html`
- OpenAPI 3 注解驱动生成，`springdoc` 配置扫描 Controller 包
- 接口按字母排序，支持中文

---

## 18. 对象转换（MapStruct）

### 18.1 设计策略

使用 MapStruct 编译期代码生成，替代运行时反射：

| 转换方向 | 示例 |
|----------|------|
| Form → Entity | `UserConverter.INSTANCE.form2Entity(userForm)` |
| Entity → VO | `UserConverter.INSTANCE.entity2Vo(user)` |
| Entity → BO | `UserConverter.INSTANCE.entity2Bo(user)` |

### 18.2 Lombok 兼容

通过 `lombok-mapstruct-binding` 确保 Lombok 与 MapStruct 的编译期注解处理器协同工作。

---

## 19. 技术栈总览

| 分类 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Java | 17 | 后端开发语言 |
| **框架** | Spring Boot | 3.3.11 | 应用框架 |
| **安全** | Spring Security | 6.x | 认证与授权 |
| **ORM** | MyBatis-Plus | 3.5.5 | 数据库操作 |
| **连接池** | Druid | 1.2.16 | 数据库连接池 |
| **缓存** | Spring Data Redis (Lettuce) | - | Redis 客户端 |
| **分布式锁/限流** | Redisson | 3.24.3 | 分布式锁、令牌桶限流 |
| **日志** | SLF4J + Logback | - | 结构化日志 |
| **认证** | Hutool JWT | 5.8.40 | JWT Token |
| **校验** | Hibernate Validator | - | 参数校验 |
| **API 文档** | Knife4j (OpenAPI 3) | 4.3.0 | Swagger API 文档 |
| **对象转换** | MapStruct | 1.5.5 | 编译期对象转换 |
| **Excel** | EasyExcel | 3.2.1 | Excel 导入导出 |
| **图片处理** | Thumbnailator | 0.4.20 | 图片压缩/缩略图 |
| **对象存储** | MinIO / 阿里云 OSS | 8.6.0 / 3.16.3 | 文件存储 |
| **WebSocket** | Spring WebSocket + STOMP | - | 实时通信 |
| **监控** | Micrometer + Prometheus | - | 指标采集 |
| **定时任务** | @Scheduled + XXL-Job | 3.3.0 | 定时任务调度 |
| **消息队列** | RabbitMQ（规划） | - | 异步任务分发 |
| **日志管道** | Kafka（规划） | - | 日志收集与流处理 |
| **测试** | JUnit 5 + TestContainers + H2 | 1.19.3 | 单元/集成测试 |
| **MongoDB** | Spring Data MongoDB | - | 文档存储（日志等） |

---

## 20. Java 与 Go 基础设施对照

| 基础设施能力 | dehaze-java | dehaze-go | 一致性 |
|-------------|-------------|-----------|--------|
| **HTTP 框架** | Spring MVC (Tomcat) | Gin | 接口语义一致 |
| **ORM** | MyBatis-Plus | GORM | 功能对等 |
| **连接池** | Druid | GORM 内置 | — |
| **缓存** | Spring Cache + Redis | 多级缓存 (FreeCache + Redis) | Java 端规划引入 L1 |
| **分布式锁** | Redisson | go-redis | 语义一致 |
| **消息队列** | 规划 RabbitMQ | 已实现 RabbitMQ | 共享 Exchange/Queue |
| **定时任务** | @Scheduled + XXL-Job | Ticker → XXL-Job | 共享 XXL-Job Admin |
| **日志** | Logback | Zap | 格式/级别统一 |
| **认证** | Spring Security + JWT | 自研中间件 + JWT | Token 格式互通 |
| **权限** | RBAC (@PreAuthorize) | RBAC (中间件) | 权限标识一致 |
| **数据权限** | MyBatis-Plus 拦截器 | GORM Scopes | 语义一致 |
| **自动填充** | MetaObjectHandler | GORM Callback | 字段名一致 |
| **API 文档** | Knife4j (OpenAPI 3) | Swagger (swag) | 规范一致 |
| **监控** | Micrometer + Prometheus | client_golang | 指标命名统一 |
| **错误码** | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | ✅ 完全一致 |
| **响应格式** | `{code, msg, data}` | `{code, msg, data}` | ✅ 完全一致 |
| **XSS 防护** | XssUtils | bluemonday | 功能对等 |
| **路径安全** | PathSecurityUtil | path_security.go | 功能对等 |
| **WebSocket** | STOMP + SockJS | — | Java 独有 |
| **MongoDB** | Spring Data MongoDB | — | Java 独有 |
