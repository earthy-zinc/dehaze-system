# Java 后端 (dehaze-java)

基于 JDK 17、Spring Boot 3.3、Spring Security 6、JWT、Redis、MyBatis-Plus、Knife4j 构建的前后端分离图像去雾系统后端。包括用户管理、角色管理、菜单管理、部门管理、字典管理等多个功能。后端自动生成接口文档，支持在线调试，提高开发效率。

> 构建/运行/测试说明见项目根目录的 `README.md`。

## 一、项目概览

### 1.1 主要功能模块

| 模块类型 | 核心实现类 | 关键技术点 |
|------|------|------|
| 安全认证 | JwtValidationFilter / SecurityConfig / SysUserDetailsService | JWT 令牌签发校验、Spring Security 鉴权 |
| 文件管理 | FileController / MinioFileService / FileUploadUtils | 适配多存储方案，支持本地/MinIO/OSS 存储，文件分片上传 |
| 系统管理 | SysController + SysServiceImpl + SysMapper | RBAC 模型、部门树形结构管理 |
| 算法管理 | SysAlgorithmController | 算法模型动态加载、Python 服务集成，对接 Python 端全部 29 种去雾算法 |
| 通用导入导出 | GenericImportExportController / ImportExportService / ExportHandlerRegistry / ImportHandlerRegistry / GenericExportStrategy / GenericImportStrategy | Handler 模式 + 通用策略，Excel/CSV 导入导出，复用 sys_task 任务框架 |
| 图像处理 | ImageUtils / FileService | 缩略图生成、EXIF 信息提取 |

### 1.2 项目难点与解决方案

1. **不同系统间通信问题:** Java 后端与 Python 算法服务的通信过程中可能导致任务阻塞，利用标准化 RESTful API 协议实现服务间通信，异步调用机制（CompletableFuture）与超时控制解决，避免服务阻塞导致超时
2. **高并发场景分布式锁竞争问题:** 高并发场景下业务处理时间与锁过期时间不匹配导致死锁，通过 @PreventDuplicateSubmit 注解组合业务 ID + 接口标识生成唯一锁 Key 解决分布式锁竞争问题
3. **大文件上传难题:** 通过文件分片上传合并策略提高大文件（>1GB）上传成功率
4. **Python 模型的内存泄漏问题:** 模型加载/卸载不及时导致内存和显存持续增长，最终会使服务器资源耗尽
5. **模型输入输出一致性问题:** 不同算法模型的输出格式统一处理，避免算法输出格式不统一导致前端解析异常
6. **算法执行超时控制与熔断机制:** 单个模型故障可能导致整个服务不可用，通过进行熔断处理，避免服务雪崩

### 1.3 项目亮点

1. **防止重复提交请求:** 通过 @PreventDuplicateSubmit 注解，利用 Spring AOP 切面注解和 Redisson 分布式锁，通过加锁并设置过期时间，防止前端请求重复提交造成损失
2. **缓存穿透防护:** 采用布隆过滤器 + 空值缓存防止大量恶意请求不存在的数据导致缓存穿透，显著降低数据库查询压力
3. **异步任务处理优化:** 采用 CompletableFuture 构建异步处理链，分为三个阶段处理去雾操作，提高系统吞吐量，降低响应延迟
4. **多存储方案适配:** 利用策略模式构建接口 FileService 实现多存储方案适配，如本地存储、MinIO、阿里云 OSS，提升系统可扩展性
5. **用户输入及权限校验:** @Validate + 自定义 @DataPermission 注解 + Mybatis 拦截器动态拼接 SQL 查询用户权限，进而利用 JWT、Spring Security 和 Redis，通过用户 ID 查询存储在 Redis 中当前用户权限，从而判断是否准许放行，后端针对传入参数通过注解验证，提供安全、无状态、分布式友好的身份验证和授权机制，提高系统健壮性
6. **三层安全防护设计:** JWT 签名验证 → Redis 权限校验 → 方法级 @DataPermission 注解，整体基于 RBAC 模型，实现细粒度的权限控制，涵盖接口方法和按钮级别
7. **项目管理:** 利用接口、枚举、泛型定义后端常量，通过继承、实现等面向对象方法统一后端响应结构体，构建全局系统异常处理器，区分开发和生产配置，提高开发效率和可维护性
8. **性能监控:** 利用 Prometheus + Grafana 通过监控指标分析系统瓶颈，优化系统性能
9. **日志管理:** 规划接入 ELK 日志系统，结合日志采集、日志分析、日志可视化等功能，提高系统日志处理效率，轻松分析日志，快速定位问题

### 1.4 接口文档

- `knife4j` 接口文档: [http://localhost:8989/doc.html](http://localhost:8989/doc.html)
- `swagger` 接口文档: [http://localhost:8989/swagger-ui/index.html](http://localhost:8989/swagger-ui/index.html)
- `apifox` 在线接口文档: [https://www.apifox.cn/apidoc](https://www.apifox.cn/apidoc/shared-195e783f-4d85-4235-a038-eec696de4ea5)

### 1.5 后续优化方案

#### 1.5.1 算法服务优化

```mermaid
sequenceDiagram
    participant SpringBoot as 后端服务
    participant PythonService as 算法微服务
    participant Nginx as 负载均衡
    participant MQ as 消息队列
    SpringBoot->>Nginx: 发送处理请求
    Nginx->>PythonService: 轮询分发
    MQ->>PythonService: 异步任务队列（可选）
    PythonService-->>SpringBoot: 返回处理结果
```

- 将 Python 算法部署为独立微服务，将系统拆分为：基础服务（Java）、算法推理服务（Python + PyTorch）、结果存储服务（Redis + MinIO），各服务可独立扩缩容
- 使用 gRPC 替代 HTTP 协议提升性能，通过 gRPC 实现算法服务负载均衡，提升高并发场景下的响应速度
- 实现模型缓存（如 ONNX Runtime 加速推理）
- 添加健康检查和熔断机制（Resilience4j），算法执行未实现熔断，单个模型故障可能导致服务雪崩

#### 1.5.2 文件存储优化

- 采用分块上传（如 MinIO 的 Part Upload）
- 增加文件元数据缓存（Redis 存储文件哈希值）

#### 1.5.3 安全性增强

- 实现 JWT 动态过期（根据用户登录时间动态计算）
- 增加 IP 黑白名单过滤
- 敏感操作二次验证（如短信/邮件验证码）

#### 1.5.4 性能监控

- 添加 Slow SQL 检测（MyBatis-Plus 插件）
- SkyWalking 链路追踪 + Prometheus 指标监控（规划中）

### 1.6 相关工程

| Gitee | Github |
|------|------|
| [dehaze-front](https://gitee.com/earthy-zinc/dehaze_front) | [dehaze-front](https://github.com/earthy-zinc/dehaze_front) |
| [dehaze-python](https://gitee.com/earthy-zinc/dehaze_python) | [dehaze-python](https://github.com/earthy-zinc/dehaze_python) |

## 二、技术基础设施

### 2.1 项目目录结构

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
│   │   │   │   ├── property/           # 配置属性类（SecurityProperties/CaptchaProperties/AlgorithmProperties/RabbitMQProperties）
│   │   │   │   ├── SecurityConfig.java # Spring Security 过滤器链
│   │   │   │   ├── MybatisConfig.java  # MyBatis-Plus 分页/数据权限/自动填充
│   │   │   │   ├── RedisConfig.java    # Redis 序列化
│   │   │   │   ├── RedisCacheConfig.java # Spring Cache + 多级缓存（Caffeine L1 + Redis L2）
│   │   │   │   ├── MultiLevelCache.java # 多级缓存接口
│   │   │   │   ├── MultiLevelCacheManager.java # 多级缓存管理器（含 Prometheus 指标）
│   │   │   │   ├── RabbitMQConfig.java # RabbitMQ Exchange/Queue/DLX/Template
│   │   │   │   ├── ResilienceConfig.java # Resilience4j 熔断器配置
│   │   │   │   ├── RestClientConfig.java # HTTP 客户端配置
│   │   │   │   ├── CorsConfig.java     # 跨域配置
│   │   │   │   ├── AsyncConfig.java    # 异步线程池（含 MDC 传播 TaskDecorator）
│   │   │   │   ├── WebMvcConfig.java   # MVC 序列化/校验
│   │   │   │   ├── WebSocketConfig.java # WebSocket STOMP
│   │   │   │   ├── WebSocketMessageRelay.java # WebSocket 消息中继
│   │   │   │   ├── SwaggerConfig.java  # Knife4j API 文档
│   │   │   │   ├── XxlJobConfig.java   # XXL-Job 执行器（条件装配）
│   │   │   │   └── CaptchaConfig.java  # 验证码
│   │   │   ├── filter/                 # Servlet 过滤器
│   │   │   │   ├── JwtValidationFilter.java
│   │   │   │   ├── TraceIdFilter.java  # TraceID 透传/生成
│   │   │   │   └── RequestLogFilter.java
│   │   │   ├── security/              # 安全组件
│   │   │   │   ├── exception/          # 认证/授权异常处理器（MyAccessDeniedHandler/MyAuthenticationEntryPoint）
│   │   │   │   ├── model/              # SysUserDetails
│   │   │   │   ├── service/            # UserDetailsService/PermissionService
│   │   │   │   └── util/               # JwtUtils/SecurityUtils
│   │   │   ├── mq/                     # 消息队列（RabbitMQ 生产者/消费者/DLX）
│   │   │   │   ├── RabbitMQPublisher.java
│   │   │   │   ├── RabbitMQConsumer.java
│   │   │   │   ├── ExportTaskConsumer.java
│   │   │   │   └── ExportDlxConsumer.java
│   │   │   ├── plugin/                 # 插件化扩展组件
│   │   │   │   ├── mybatis/            # MyBatis 插件（数据权限/自动填充）
│   │   │   │   ├── dupsubmit/          # 防重复提交（AOP + Redisson）
│   │   │   │   ├── ratelimit/          # 接口限流（AOP + Redisson）
│   │   │   │   ├── captcha/            # 验证码配置属性
│   │   │   │   └── easyexcel/          # Excel 导入监听器
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
│           ├── db/                     # 测试 SQL（init.sql + h2/，schema/ 和 data/ 目录由 Maven 从 config/sql/ 自动复制）
│           └── templates/              # 代码生成模板
```

### 2.2 分层架构设计

#### 2.2.1 架构分层

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

#### 2.2.2 层级职责

| 层级 | 包路径 | 职责 | 依赖方向 |
|------|--------|------|----------|
| **Filter 链** | `filter/` + Spring Security | 请求拦截、JWT 校验、验证码、跨域 | ← 外部请求 |
| **Controller 层** | `controller/` | 参数绑定与校验、调用 Service、统一响应 | → Service |
| **Service 层** | `service/` | 业务逻辑编排、事务边界、缓存交互 | → Mapper + plugin |
| **Mapper 层** | `mapper/` | 数据库 CRUD、SQL 构建、数据权限 | → MyBatis-Plus |
| **基础设施层** | `config/` + `plugin/` + `common/` | 配置、缓存、安全、限流等基础能力 | 被所有层依赖 |

#### 2.2.3 依赖注入策略

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

#### 2.2.4 数据模型分层

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

### 2.3 配置管理

#### 2.3.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 配置加载 | Spring Boot YAML | 多 Profile、环境变量覆盖 |
| 属性绑定 | `@ConfigurationProperties` | 类型安全的配置绑定 |
| 环境变量 | `${ENV_VAR}` 占位符 | 敏感信息外部化 |
| 参数校验 | Hibernate Validator | JSR-380 Bean Validation |

#### 2.3.2 配置结构

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

#### 2.3.3 多环境支持

| 环境 | Profile | 配置文件 | 特性差异 |
|------|---------|----------|----------|
| **开发** | dev | `application-dev.yml` | SQL 日志输出、Swagger 启用、DevTools 热重载 |
| **测试** | test | `application-test.yml` | H2 内存数据库 / TestContainers、缓存禁用 |
| **生产** | prod | `application-prod.yml` | Swagger 禁用、连接池优化、日志输出到文件 |

#### 2.3.4 敏感信息管理

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

#### 2.3.5 条件化装配

通过 `@ConditionalOnProperty` 按配置开关控制组件加载：

| 组件 | 配置项 | 说明 |
|------|--------|------|
| XXL-Job | `xxl.job.enabled=true` | 定时任务执行器按需加载 |
| Redis Cache | `spring.cache.enabled=true` | Spring Cache 按需启用 |

### 2.4 数据访问层

#### 2.4.1 技术选型

| 组件 | 选型 | 版本 | 说明 |
|------|------|------|------|
| ORM | MyBatis-Plus | 3.5.5 | 通用 CRUD、分页、数据权限 |
| 连接池 | Druid | 1.2.16 | 监控、防 SQL 注入、连接管理 |
| 数据库 | MySQL | 9.5.0 (驱动) | 生产环境 |
| 测试数据库 | H2 / TestContainers(MySQL) | - | 单元测试 / 集成测试 |

#### 2.4.2 连接池配置

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

#### 2.4.3 MyBatis-Plus 插件链

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

#### 2.4.4 自动填充（MetaObjectHandler）

| 触发时机 | 填充字段 | 值来源 |
|----------|----------|--------|
| INSERT | `createTime` / `updateTime` | `LocalDateTime.now()` |
| INSERT | `createBy` / `updateBy` | `SecurityUtils.getUserId()` |
| UPDATE | `updateTime` | `LocalDateTime.now()` |
| UPDATE | `updateBy` | `SecurityUtils.getUserId()` |

#### 2.4.5 数据权限（DataScope）

基于 MyBatis-Plus `DataPermissionHandler` 实现行级数据权限控制，通过 `@DataPermission` 注解标记需要权限控制的 Mapper 方法：

| 权限范围 | 枚举值 | SQL 效果 |
|----------|--------|----------|
| 全部数据 | `ALL` | 无附加条件 |
| 本部门数据 | `DEPT` | `WHERE dept_id = ?` |
| 本部门及下级 | `DEPT_AND_SUB` | `WHERE dept_id IN (SELECT id FROM sys_dept WHERE ...)` |
| 仅本人数据 | `SELF` | `WHERE create_by = ?` |

#### 2.4.6 逻辑删除

全局配置逻辑删除：

```yaml
mybatis-plus:
  global-config:
    db-config:
      logic-delete-field: deleted
      logic-delete-value: 1
      logic-not-delete-value: 0
```

### 2.5 缓存体系

#### 2.5.1 架构设计

```mermaid
flowchart TB
    subgraph CacheArch["缓存体系"]
        direction TB

        subgraph SpringCache["Spring Cache (注解式)"]
            Annotation["@Cacheable / @CacheEvict<br/>声明式缓存"]
        end

        subgraph MultiLevel["多级缓存 (MultiLevelCacheManager)"]
            L1["L1 Caffeine<br/>本地缓存（5min TTL）"]
            L2["L2 Redis<br/>分布式缓存（1h TTL）"]
            Metrics["Prometheus 指标<br/>hits/misses/loader"]
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

    SpringCache --> MultiLevel
    L1 --> L2
    L2 --> Redis
    RedisTemplate --> Redis
    Redisson --> Redis
```

#### 2.5.2 Spring Cache 配置

| 配置项 | 值 | 说明 |
|--------|------|------|
| 后端类型 | Multi-Level (Caffeine + Redis) | `@Primary cacheManager = MultiLevelCacheManager` |
| L1 TTL | 5min | Caffeine 本地缓存 |
| L2 TTL | 3600s | Redis 分布式缓存 |
| 空值缓存 | 启用 | `allowNullValues=true`，防缓存穿透 |
| Key 序列化 | StringRedisSerializer | 可读性 |
| Value 序列化 | GenericJackson2JsonRedisSerializer | JSON 格式，支持 JSR-310 时间类型 |
| Key 前缀 | `cacheName:` | 覆盖默认双冒号分隔符 |
| 开关 | `spring.cache.enabled=true` | `@ConditionalOnProperty` 控制启用 |

#### 2.5.3 Redis 序列化

`RedisTemplate<String, Object>` 自定义序列化，避免 JDK 默认序列化的可读性问题：

```
Key:   StringRedisSerializer (UTF-8 字符串)
Value: GenericJackson2JsonRedisSerializer (JSON，含类型信息)
Hash Key:   StringRedisSerializer
Hash Value: GenericJackson2JsonRedisSerializer
```

#### 2.5.4 分布式锁（Redisson）

| 使用场景 | 实现方式 | 说明 |
|----------|----------|------|
| 防重复提交 | `RLock` + `tryLock` | 基于 Token JTI + 请求路径生成锁 Key |
| 接口限流 | `RRateLimiter` | 基于 Redisson 令牌桶算法 |

#### 2.5.5 多级缓存指标

`MultiLevelCacheManager` 内置 Prometheus 指标采集：

| 指标 | 类型 | 说明 |
|------|------|------|
| `dehaze_cache_hits_total` | Counter | 缓存命中次数（含 L1/L2 标签） |
| `dehaze_cache_misses_total` | Counter | 缓存未命中次数（含 L1/L2 标签） |
| `dehaze_cache_loader_total` | Counter | 回源加载次数 |

#### 2.5.6 缓存使用情况

Spring Cache 注解使用位置（共 10 处）：

| 模块 | 缓存名 | 注解 |
|------|--------|------|
| SysMenuServiceImpl | `menu` | `@Cacheable(key='options'\|'routes')` + `@CacheEvict` |
| SysDatasetServiceImpl | `dataset:all` / `dataset:statsMap` / `dataset:options` | `@Cacheable` + `@CacheEvict(allEntries=true)` |
| SysRoleServiceImpl | `menu` | `@CacheEvict(key='routes')` |

#### 2.5.7 缓存演进规划

```
当前: Spring Cache → Caffeine (L1) + Redis (L2) 多级缓存
规划: + 布隆过滤器 (穿透防护，使用 Redisson RBloomFilter)
      + SingleFlight (击穿防护)
      + Pub/Sub (多实例 L1 失效广播)
```

### 2.6 消息队列

#### 2.6.1 技术选型

与 Go 端保持一致的中间件选型：

| 消息中间件 | 用途 | 当前状态 |
|------------|------|----------|
| **RabbitMQ** | 异步任务分发（导出、批量操作等） | ✅ 已实现（`spring-boot-starter-amqp`） |
| **Kafka** | 日志收集与流处理 | 📋 规划中（`spring-kafka` 死依赖待清理或激活） |

#### 2.6.2 双通道任务分发架构

采用 **MQ 消费者 + @Async 线程池** 双通道架构。`TaskExecutorImpl` 通过 `@Async` 直接执行策略，同时 `ExportTaskConsumer` 监听 RabbitMQ 队列（当前消费者实现仍为 TODO 桩，待与 @Async 路径合并为单一 MQ 路径）：

```mermaid
flowchart LR
    subgraph TaskSystem["异步任务系统"]
        Controller["TaskController"] --> TaskService
        TaskService --> TaskExecutor["TaskExecutorImpl<br/>@Async(datasetTaskExecutor)"]
        TaskService --> Publisher["RabbitMQPublisher"]
        TaskExecutor --> Factory["TaskStrategyFactory"]
        Factory --> S1["DatasetExportStrategy"]
        Factory --> S2["BatchDownloadStrategy"]
        Factory --> S3["CustomExportStrategy"]
        Factory --> S4["ItemDownloadStrategy"]
    end

    subgraph MQ["RabbitMQ Broker"]
        Exchange["dehaze.tasks<br/>(direct exchange)"]
        Q1["task.export<br/>导出任务队列"]
        DLX["Dead Letter Exchange<br/>死信交换机"]
    end

    subgraph Consumer["消费者 (dehaze-java)"]
        ExportConsumer["ExportTaskConsumer<br/>@RabbitListener"]
        DlxConsumer["ExportDlxConsumer<br/>@RabbitListener(DLX)"]
    end

    subgraph Progress["进度回调"]
        WebSocket["WebSocket STOMP"]
    end

    Publisher --> Exchange --> Q1 --> ExportConsumer
    Q1 -.->|"nack/超时"| DLX --> DlxConsumer
    TaskExecutor -.->|"ProgressCallback"| WebSocket
```

#### 2.6.3 线程池配置

`@Async` 使用的线程池配置：

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

#### 2.6.4 RabbitMQ 配置

`RabbitMQConfig.java` 中声明的拓扑：

| 组件 | 名称 | 说明 |
|------|------|------|
| Exchange | `dehaze.tasks` (direct) | 与 Go/Python 端一致 |
| Queue | `task.export` | 导出任务队列（durable, TTL=24h） |
| DLX Exchange | 死信交换机 | nack/超时消息转入 |
| DLX Queue | `task.export.dlx` | 死信队列，由 `ExportDlxConsumer` 消费 |

#### 2.6.5 已知问题

| 问题 | 严重级别 | 概述 |
|------|----------|------|
| **MQ 消费者为 TODO 桩** | P0 | `ExportTaskConsumer` 收到消息后未实际调用策略，需补全或将 `@RabbitListener` 改为调用 `TaskExecutor.submitExportTask` |
| **双路径死代码** | P1 | `@Async` 与 MQ 并存，MQ 路径从未被真正调用 |
| **Kafka 死依赖** | P2 | `pom.xml` 引入 `spring-kafka` 但无任何 `@KafkaListener`/`KafkaTemplate` 使用 |
| **MongoDB 死依赖** | P2 | `pom.xml` 引入 `spring-boot-starter-data-mongodb` 但无 `@Document`/`MongoTemplate` 使用 |

#### 2.6.6 Kafka 规划（日志管道）

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

### 2.7 通用导入导出框架

系统提供统一的导入导出能力，通过 **Handler 模式 + 通用策略** 实现复用，各业务模块只需实现 `ExportHandler`/`ImportHandler` 接口，不各自编写 Controller/Service。

#### 2.7.1 核心组件

| 组件 | 职责 |
|------|------|
| `GenericImportExportController` | 统一入口，提供 `/{module}/_export`、`/{module}/_import`、`/{module}/template` 接口 |
| `ImportExportService` | 通用服务层：同步/异步判断、文件验证、Handler 路由、任务创建 |
| `ExportHandlerRegistry` / `ImportHandlerRegistry` | 处理器注册表，启动时按 `getModule()` 自动注册 |
| `TemplateManager` | 模板动态生成，根据 `getFieldConfigs()` 生成表头和示例数据 |
| `ImportExportFileGenerator` | 文件生成器，封装 EasyExcel（Excel）和 Apache Commons CSV（CSV）流式写入 |
| `GenericExportStrategy` / `GenericImportStrategy` | 通用任务策略，注册到 `TaskStrategyFactory`，处理所有 `xxx_export`/`xxx_import` 任务类型 |

#### 2.7.2 Handler 接口

- **ExportHandler**：`getModule()`、`estimateCount()`、`export()`、`getFieldConfigs()`
- **ImportHandler**：`getModule()`、`getFieldConfigs()`、`importBatch()`、`getTemplateSampleData()`

#### 2.7.3 已实现的处理器

| 模块 | ExportHandler | ImportHandler |
|------|--------------|--------------|
| 用户管理 | UserExportHandler | UserImportHandler |
| 角色管理 | RoleExportHandler | RoleImportHandler |
| 部门管理 | DeptExportHandler | DeptImportHandler |
| 菜单管理 | MenuExportHandler | MenuImportHandler |
| 字典管理 | DictExportHandler | DictImportHandler |
| 数据集管理 | DatasetExportHandler | -（仅导出） |
| 算法管理 | AlgorithmExportHandler | AlgorithmImportHandler |

> 详细的接口设计、处理器实现、三端对齐要点详见 [任务管理/后端实现.md](../../03-模块设计/基础模块/任务管理/后端实现.md)。

### 2.8 定时任务

#### 2.8.1 当前方案：Spring @Scheduled（XXL-Job 仅集成框架）

| 调度方式 | 适用场景 | 当前状态 |
|----------|----------|----------|
| `@Scheduled` | 轻量级、单实例定时任务 | ✅ 已启用 |
| XXL-Job | 分布式调度、Web 管理 | ⚠️ 仅集成 `XxlJobSpringExecutor` Bean，**未注册任何 `@XxlJob` Handler** |

#### 2.7.2 内置定时任务（@Scheduled）

| 任务 | CRON | 功能 |
|------|------|------|
| `cleanupExpiredTasks` | `0 0 2 * * ?` | 每天凌晨 2 点清理 7 天前已完成/取消任务、30 天前所有任务 |
| `cleanupStuckTasks` | `0 0 * * * ?` | 每小时清理超过 24 小时的异常状态任务（PROCESSING 超 30min / PENDING 超 24h 标记 FAILED） |

#### 2.7.3 XXL-Job 集成

通过 `@ConditionalOnProperty(name = "xxl.job.enabled")` 按需装配 `XxlJobSpringExecutor`：

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
        Bean["XxlJobSpringExecutor<br/>(仅注册 Bean)"]
        Note["⚠️ 无 @XxlJob Handler<br/>待迁移 @Scheduled 任务"]
    end

    Scheduler -.->|"预留 HTTP 回调通道"| JavaExecutor
    JavaExecutor -.->|"预留结果上报"| Scheduler
```

#### 2.7.4 迁移计划

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | XXL-Job Admin 部署（与 Go/Python 端共享） | ✅ 已完成 |
| **Phase 2** | `XxlJobSpringExecutor` Bean 集成 | ✅ 已完成 |
| **Phase 3** | 将 `@Scheduled` 任务迁移到 XXL-Job（注册 `@XxlJob` Handler） | 📋 规划中 |
| **Phase 4** | 新增统计/维护类任务直接注册到 XXL-Job | 📋 规划中 |

### 2.8 安全过滤器链

#### 2.8.1 HTTP 服务与过滤器链

**技术选型**

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | Spring Boot 3 + Spring MVC | 内嵌 Tomcat |
| API 文档 | Knife4j (OpenAPI 3) | 开发环境自动挂载 |
| WebSocket | Spring WebSocket + STOMP | 实时消息推送 |

**过滤器链**

请求经过的过滤器按注册顺序执行：

```mermaid
flowchart LR
    Req["请求"] --> CORS["跨域处理<br/>CorsFilter<br/>order=-101"]
    CORS --> Trace["TraceID 透传<br/>TraceIdFilter"]
    Trace --> JWT["JWT 校验<br/>JwtValidationFilter"]
    JWT --> Security["Spring Security<br/>FilterChain"]
    Security --> Permission["权限校验<br/>@PreAuthorize"]
    Permission --> Handler["业务处理<br/>Controller"]
```

> 注：验证码校验逻辑已合并到登录流程内（不再使用独立 `CaptchaValidationFilter`）。

**过滤器清单**

| 过滤器 | 文件 | 功能 | 作用范围 |
|--------|------|------|----------|
| CorsFilter | `CorsConfig.java` | 跨域资源共享，order=-101（先于 Security） | 全局 |
| TraceIdFilter | `filter/TraceIdFilter.java` | TraceID 生成/透传/回写 MDC | 全局 |
| JwtValidationFilter | `filter/JwtValidationFilter.java` | JWT Token 验证、SecurityContext 注入 | 受保护路由 |
| SecurityFilterChain | `SecurityConfig.java` | Spring Security 认证/授权链 | 全局 |
| RequestLogFilter | `filter/RequestLogFilter.java` | 请求 URI 日志 | 全局 |

**Spring Security 过滤器链配置**

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

#### 2.8.2 安全基础设施

**认证体系**

| 组件 | 实现 | 说明 |
|------|------|------|
| JWT | Hutool JWT | AccessToken（`security.jwt.ttl` 秒有效期） |
| Token 黑名单 | Redis `BLACKLIST_TOKEN:` | 注销时将 JTI 加入黑名单 |
| 密码加密 | BCryptPasswordEncoder | Spring Security 标准实现 |
| 验证码 | Hutool Captcha | 支持圆圈/GIF/干扰线/扭曲多种类型 |
| UserDetails | SysUserDetailsService | 从数据库加载用户信息 |

**权限体系**

RBAC 权限模型通过 Spring Security `@PreAuthorize` + 自定义 `PermissionService` 实现：

```
用户 → 角色（多对多） → 权限标识（多对多）
权限格式: 模块:功能:操作（如 sys:user:add）
```

**权限校验流程**：

```mermaid
flowchart LR
    A["@PreAuthorize<br/>@ss.hasPerm('sys:user:add')"] --> B["PermissionService.hasPerm()"]
    B --> C{"超级管理员?"}
    C -->|是| D["直接放行"]
    C -->|否| E["从 Redis 获取角色权限"]
    E --> F{"包含所需权限?"}
    F -->|是| G["放行"]
    F -->|否| H["拒绝 403"]
```

权限缓存存储在 Redis Hash：`ROLE_PERMS_PREFIX → {roleCode: Set<String> perms}`

**安全工具**

| 工具 | 文件 | 功能 |
|------|------|------|
| XSS 过滤 | `util/XssUtils.java` | HTML 标签清洗 |
| 路径安全 | `util/PathSecurityUtil.java` | 路径穿越检测、安全路径拼接 |
| 安全上下文 | `security/util/SecurityUtils.java` | 获取当前用户 ID/部门/角色/数据权限 |
| JWT 工具 | `security/util/JwtUtils.java` | Token 解析、Authentication 构建 |

#### 2.8.3 插件化扩展组件

**设计理念**

通过 `plugin/` 包实现可插拔的横切关注点，每个插件独立封装为注解 + AOP 切面：

| 插件 | 包路径 | 注解 | 实现方式 |
|------|--------|------|----------|
| 防重复提交 | `plugin/dupsubmit/` | `@PreventDuplicateSubmit` | AOP + Redisson 分布式锁 |
| 接口限流 | `plugin/ratelimit/` | `@RateLimit` | AOP + Redisson 令牌桶 |
| 数据权限 | `plugin/mybatis/` | `@DataPermission` | MyBatis-Plus 拦截器 |
| 字段自动填充 | `plugin/mybatis/` | `@TableField(fill=...)` | MetaObjectHandler |

**防重复提交**

```mermaid
flowchart LR
    A["@PreventDuplicateSubmit<br/>expire=5s"] --> B["DuplicateSubmitAspect"]
    B --> C["生成锁 Key<br/>LOCK:RESUBMIT:{jti}:{method}-{uri}"]
    C --> D["Redisson tryLock"]
    D -->|获取成功| E["执行业务"]
    D -->|获取失败| F["抛出 REPEAT_SUBMIT_ERROR"]
    E --> G["释放锁"]
```

**接口限流**

```mermaid
flowchart LR
    A["@RateLimit<br/>type=IP, maxRequests=10, timeWindow=60"] --> B["RateLimitAspect"]
    B --> C["构建限流 Key<br/>rateLimit:{ip}:{class}#{method}"]
    C --> D["Redisson RRateLimiter"]
    D -->|获取令牌| E["执行业务"]
    D -->|令牌不足| F["抛出 RateLimitException"]
```

限流维度支持：IP / USER / GLOBAL。

### 2.9 日志系统

#### 2.9.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 日志框架 | SLF4J + Logback | SpringBoot 默认日志实现 |
| 日志切割 | TimeBasedRollingPolicy | 按天切割 + 按大小分片（10MB） |
| 日志保留 | maxHistory=15 | 保留 15 天 |

#### 2.9.2 日志配置

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

#### 2.9.3 日志分层

| 层级 | 日志内容 | 示例 |
|------|----------|------|
| **Filter 层** | 请求 URI 日志 | `request uri: /api/v1/users` |
| **Controller 层** | Spring MVC 请求日志 | 由 RequestLogFilter 记录 |
| **Service 层** | 业务关键节点 | `用户登录成功 userId=123` |
| **Mapper 层** | SQL 日志（开发环境） | `==> Preparing: SELECT * FROM sys_user WHERE id = ?` |
| **异常处理** | 全局异常日志 | `biz exception: 用户不存在` |

#### 2.9.4 日志演进规划

```
当前: SLF4J/Logback → 文件（按天切割）
规划: Logback → Kafka → Elasticsearch（检索）/ 对象存储（归档）
```

### 2.10 可观测性

#### 2.10.1 指标采集（Metrics）

| 指标类型 | 实现方式 | 端点 |
|----------|----------|------|
| JVM 指标 | Micrometer + Prometheus | `/actuator/prometheus` |
| HTTP 请求指标 | Spring Boot Actuator | `/actuator/prometheus` |
| 自定义业务指标 | Micrometer API | `/actuator/prometheus` |

#### 2.10.2 健康检查

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

#### 2.10.3 API 文档

- 开发环境自动挂载 Knife4j UI：`GET /doc.html`
- OpenAPI 3 注解驱动生成，`springdoc` 配置扫描 Controller 包
- 接口按字母排序，支持中文

### 2.11 三端对照

| 基础设施能力 | dehaze-java | dehaze-go | 一致性 |
|-------------|-------------|-----------|--------|
| **HTTP 框架** | Spring MVC (Tomcat) | Gin | 接口语义一致 |
| **ORM** | MyBatis-Plus | GORM | 功能对等 |
| **连接池** | Druid | GORM 内置 | — |
| **缓存** | Spring Cache + Caffeine L1 + Redis L2 | 多级缓存 (gokit local + Redis) | 已对齐多级缓存 |
| **分布式锁** | Redisson | go-redis | 语义一致 |
| **消息队列** | ✅ RabbitMQ（消费者 TODO 桩） | ✅ RabbitMQ（已实现） | 共享 Exchange/Queue |
| **定时任务** | @Scheduled + XXL-Job（仅 Executor） | Ticker + XXL-Job（规划） | 共享 XXL-Job Admin |
| **日志** | Logback | Zap | 格式/级别统一 |
| **认证** | Spring Security + JWT | 自研中间件 + JWT | Token 格式互通 |
| **权限** | RBAC (@PreAuthorize) | RBAC (中间件) | 权限标识一致 |
| **数据权限** | MyBatis-Plus 拦截器 | GORM Plugin (DataScopePlugin) | 语义一致 |
| **自动填充** | MetaObjectHandler | GORM Callback | 字段名一致 |
| **API 文档** | Knife4j (OpenAPI 3) | Swagger (swag) | 规范一致 |
| **监控** | Micrometer + Prometheus | client_golang | 指标命名统一 |
| **错误码** | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | ✅ 完全一致 |
| **响应格式** | `{code, msg, data}` | `{code, msg, data}` | ✅ 完全一致 |
| **XSS 防护** | XssUtils | bluemonday | 功能对等 |
| **路径安全** | PathSecurityUtil | path_security.go | 功能对等 |
| **WebSocket** | STOMP + SockJS | — | Java 独有 |
| **TraceID** | TraceIdFilter + MDC | trace.go + Context | 语义一致 |

### 2.12 应用生命周期管理

#### 2.12.1 启动流程

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

#### 2.12.2 优雅关闭

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

#### 2.12.3 多 Profile 配置

通过 `spring.profiles.active` 控制环境切换：

```yaml
# application.yml
spring:
  profiles:
    active: dev  # 可选: dev / prod / test
```

### 2.13 统一响应与错误处理

#### 2.13.1 响应格式

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

#### 2.13.2 错误码体系

错误码采用 **5 位字符串** 编码，与 Go 端保持一致：

| 前缀 | 类别 | 示例 |
|------|------|------|
| `00` | 成功 | `00000` - 一切 ok |
| `A0` | 用户端错误 | `A0001` - 用户端错误, `A0200` - 登录异常, `A0400` - 参数错误 |
| `B0` | 系统执行错误 | `B0001` - 系统执行出错, `B0210` - 并发限流 |
| `C0` | 第三方服务错误 | `C0001` - 调用第三方服务出错, `C0300` - 数据库服务出错 |

#### 2.13.3 全局异常处理

通过 `@RestControllerAdvice` + `@ExceptionHandler` 统一拦截并格式化输出，共处理 18 类异常：

```mermaid
flowchart TB
    E1["BindException<br/>ConstraintViolationException<br/>MethodArgumentNotValidException<br/>MissingServletRequestParameterException<br/>MethodArgumentTypeMismatchException<br/>TypeMismatchException<br/>JsonProcessingException<br/>HttpMessageNotReadableException<br/>IllegalArgumentException<br/>ServletException"] -->|400| R1["ResultCode.PARAM_ERROR"]
    E2["BusinessException"] -->|400| R2["自定义 ResultCode"]
    E3["RateLimitException"] -->|429| R3["ResultCode.RATE_LIMIT"]
    E4["NoHandlerFoundException"] -->|404| R4["ResultCode.RESOURCE_NOT_FOUND"]
    E5["DuplicateKeyException<br/>DataIntegrityViolationException"] -->|400| R5["唯一键/数据完整性冲突"]
    E6["BadSqlGrammarException<br/>SQLSyntaxErrorException"] -->|500| R6["SQL 语法错误"]
    E7["AccessDeniedException<br/>AuthenticationException"] -->|"继续抛出"| R7["Security 异常处理器<br/>MyAccessDeniedHandler / MyAuthenticationEntryPoint"]
    E8["Exception (兜底)"] -->|500| R8["SYSTEM_EXECUTION_ERROR"]
```

**异常分类处理**：

| 异常类型 | HTTP 状态码 | 处理方式 |
|----------|-------------|----------|
| 参数校验异常（含类型转换/JSON 解析/缺失参数等 10 类） | 400 | 收集校验错误信息，拼接返回 |
| 业务异常 | 400 | 返回 BusinessException 中的 ResultCode |
| 限流异常 | 429 | 返回 RATE_LIMIT 错误码 |
| 资源不存在 | 404 | 返回 RESOURCE_NOT_FOUND |
| 唯一键冲突 / 数据完整性 | 400 | 返回数据库冲突信息 |
| SQL 语法错误 | 500 | 返回系统执行错误 |
| Security 异常（`AccessDeniedException` / `AuthenticationException`） | - | 兜底处理器内 `instanceof` 判断后**重新抛出**，由 `MyAccessDeniedHandler`/`MyAuthenticationEntryPoint` 处理 |
| 未知异常 | 500 | 兜底处理，记录日志 |

### 2.14 参数校验

#### 2.14.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 校验框架 | Hibernate Validator (JSR-380) | Bean Validation 标准实现 |
| FailFast | 启用 | 遇到第一个校验失败立即返回 |
| 自定义校验 | `@FileExists` / `@DirExists` | 文件/目录存在性校验 |

#### 2.14.2 校验策略

- Form 对象使用 `@NotBlank` / `@NotNull` / `@Size` / `@Pattern` 等注解声明校验规则
- Controller 使用 `@Valid` / `@Validated` 触发校验
- FailFast 模式：第一个校验失败立即返回，避免收集过多错误信息

#### 2.14.3 自定义校验注解

| 注解 | 校验器 | 功能 |
|------|--------|------|
| `@FileExists` | `FileExistValidator` | 校验文件路径是否存在 |
| `@DirExists` | `DirExistValidator` | 校验目录路径是否存在 |

### 2.15 WebSocket 实时通信

#### 2.15.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 协议 | STOMP over WebSocket | 消息语义化 |
| 兼容 | SockJS | 浏览器不支持 WebSocket 时降级 |
| 认证 | JWT Token 解析 | CONNECT 帧中提取用户信息 |

#### 2.15.2 端点配置

| 端点 | 协议 | 说明 |
|------|------|------|
| `/ws` | WebSocket + SockJS | 浏览器客户端 |
| `/ws-app` | 原生 WebSocket | uni-app 客户端 |

#### 2.15.3 消息代理

```
客户端发送前缀: /app
订阅前缀: /topic (广播) / /queue (点对点)
用户前缀: /user
```

### 2.16 对象转换（MapStruct）

#### 2.16.1 设计策略

使用 MapStruct 编译期代码生成，替代运行时反射：

| 转换方向 | 示例 |
|----------|------|
| Form → Entity | `UserConverter.INSTANCE.form2Entity(userForm)` |
| Entity → VO | `UserConverter.INSTANCE.entity2Vo(user)` |
| Entity → BO | `UserConverter.INSTANCE.entity2Bo(user)` |

#### 2.16.2 Lombok 兼容

通过 `lombok-mapstruct-binding` 确保 Lombok 与 MapStruct 的编译期注解处理器协同工作。

### 2.17 技术栈总览

| 分类 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Java | 17 | 后端开发语言 |
| **框架** | Spring Boot | 3.3.11 | 应用框架 |
| **安全** | Spring Security | 6.x | 认证与授权 |
| **ORM** | MyBatis-Plus | 3.5.5 | 数据库操作 |
| **连接池** | Druid | 1.2.16 | 数据库连接池 |
| **缓存** | Spring Data Redis (Lettuce) + Caffeine | - | Redis 客户端 + L1 本地缓存 |
| **分布式锁/限流** | Redisson | 3.24.3 | 分布式锁、令牌桶限流 |
| **熔断器** | Resilience4j | 2.2.0 | 熔断/重试/限流 |
| **日志** | SLF4J + Logback + logstash-logback-encoder | 7.4 | 结构化日志（含 JSON 编码器） |
| **认证** | Hutool JWT | 5.8.40 | JWT Token |
| **校验** | Hibernate Validator | - | 参数校验 |
| **API 文档** | Knife4j (OpenAPI 3) | 4.3.0 | Swagger API 文档 |
| **对象转换** | MapStruct | 1.5.5.Final | 编译期对象转换 |
| **Excel** | EasyExcel | 3.2.1 | Excel 导入导出 |
| **图片处理** | Thumbnailator | 0.4.20 | 图片压缩/缩略图 |
| **对象存储** | MinIO / 阿里云 OSS | 8.6.0 / 3.16.3 | 文件存储 |
| **WebSocket** | Spring WebSocket + STOMP | - | 实时通信 |
| **监控** | Micrometer + Prometheus | - | 指标采集 |
| **定时任务** | @Scheduled + XXL-Job (Executor) | 3.3.0 | 定时任务调度（XXL-Job Handler 未注册） |
| **消息队列** | RabbitMQ (spring-boot-starter-amqp) | - | 异步任务分发（消费者 TODO 桩） |
| **日志管道** | Kafka（死依赖，规划激活） | - | 日志收集与流处理 |
| **HTTP 客户端** | Apache HttpClient 5 | - | 调用 Python 算法服务 |
| **测试** | JUnit 5 + TestContainers + H2 | 1.19.3 / 2.2.224 | 单元/集成测试 |
| **Lombok** | Lombok + lombok-mapstruct-binding | 1.18.32 / 0.2.0 | 样板代码消除 |

> ⚠️ **死依赖提示**：`spring-boot-starter-data-mongodb` 和 `spring-kafka` 在 `pom.xml` 中声明但代码中无任何使用，建议清理或激活。
