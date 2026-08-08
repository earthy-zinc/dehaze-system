# Java 后端 (dehaze-java)

基于 JDK 17、Spring Boot 3.3、Spring Security 6、Redis、MyBatis-Plus 构建的前后端分离图像去雾系统后端，涵盖用户管理、角色管理、菜单管理、部门管理、字典管理等功能模块。

> 构建/运行/测试说明见项目根目录的 `README.md`。

## 一、分层架构

```mermaid
flowchart TB
    subgraph External["外部请求"]
        Client["HTTP Client"]
    end

    subgraph FilterChain["Servlet Filter 链"]
        direction LR
        Trace["TraceIdFilter"]
        CORS["跨域 CorsFilter"]
        Log["RequestLogFilter"]
        ApiKey["ApiKeyAuthenticationFilter"]
        Session["SessionFilter 会话校验"]
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

    subgraph Infrastructure["基础设施层 (config/ + plugin/ + common/)"]
        direction LR
        DB[("MySQL")]
        Cache[("Redis")]
        MQ[("RabbitMQ")]
    end

    Client --> FilterChain --> Controller --> Service --> Mapper --> Infrastructure
```

### 层级职责

| 层级 | 包路径 | 职责 | 依赖方向 |
|------|--------|------|----------|
| Filter 链 | `filter/` + Spring Security | 请求拦截、Session 校验、验证码、跨域 | 外部请求 |
| Controller 层 | `controller/` | 参数绑定与校验、调用 Service、统一响应 | -> Service |
| Service 层 | `service/` | 业务逻辑编排、事务边界、缓存交互 | -> Mapper + plugin |
| Mapper 层 | `mapper/` | 数据库 CRUD、SQL 构建、数据权限 | -> MyBatis-Plus |
| 基础设施层 | `config/` + `plugin/` + `common/` | 配置、缓存、安全、限流等基础能力 | 被所有层依赖 |

### 依赖注入策略

基于 Spring IoC 容器实现自动依赖装配：

- 使用 `@RequiredArgsConstructor` (Lombok) 生成构造函数注入，优先于字段注入
- 配置类使用 `@Configuration` + `@Bean` 显式声明基础组件
- 条件装配使用 `@ConditionalOnProperty` 控制组件按需加载（如 XXL-Job、Redis Cache）
- 属性绑定使用 `@ConfigurationProperties` + `@ConfigurationPropertiesScan`

### 数据模型分层

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

| 模型类型 | 包路径 | 职责 | 定位 |
|----------|--------|------|------|
| Entity | `model/entity/` | 数据库表映射，MyBatis-Plus 注解 | 主力模型 |
| Form | `model/form/` | 请求入参绑定、校验注解 | 主力模型 |
| VO | `model/vo/` | API 响应输出 | 主力模型 |
| Query | `model/query/` | 分页查询条件 | 主力模型 |
| BO | `model/bo/` | 业务层跨服务聚合传递 | 按需使用 |
| DTO | `model/dto/` | 跨层数据传输（如登录结果） | 按需使用 |
| Event | `model/event/` | 领域事件载荷 | 按需使用 |

对象转换使用 MapStruct 编译期生成转换代码（`converter/` 包），避免运行时反射开销。

## 二、项目目录结构

```
dehaze-java/
├── pom.xml                             # Maven 依赖管理
├── src/
│   ├── main/
│   │   ├── java/com/pei/dehaze/
│   │   │   ├── SystemApplication.java  # SpringBoot 启动入口
│   │   │   ├── common/                 # 公共基础模块
│   │   │   │   ├── base/               # 基类（BaseEntity/BasePageQuery/IBaseEnum）
│   │   │   │   ├── constant/           # 常量定义（Security/Session/Task）
│   │   │   │   ├── enums/              # 业务枚举（状态/类型/权限范围）
│   │   │   │   ├── exception/          # 异常体系（BusinessException + 全局处理器）
│   │   │   │   ├── model/              # 公共模型（Option）
│   │   │   │   ├── result/             # 统一响应（Result/ResultCode/PageResult）
│   │   │   │   ├── util/               # 工具类（XSS/路径安全/文件/日期）
│   │   │   │   └── validator/          # 自定义校验注解
│   │   │   ├── config/                 # 配置类（Security/Mybatis/Redis/Cache/MQ/Resilience/WebSocket等）
│   │   │   ├── filter/                 # Servlet 过滤器（TraceId/RequestLog/JwtValidation）
│   │   │   ├── security/              # 安全组件（认证/授权/工具）
│   │   │   ├── mq/                     # 消息队列（RabbitMQ 生产者/消费者/DLX）
│   │   │   ├── plugin/                 # 插件化扩展组件
│   │   │   │   ├── mybatis/            # MyBatis 插件（数据权限/自动填充）
│   │   │   │   ├── dupsubmit/          # 防重复提交（AOP + Redisson）
│   │   │   │   ├── ratelimit/          # 接口限流（AOP + Redisson）
│   │   │   │   └── easyexcel/          # Excel 导入监听器
│   │   │   ├── controller/             # Controller 层
│   │   │   ├── service/                # Service 层（含 file/ 存储策略实现、strategy/ 任务策略）
│   │   │   ├── mapper/                 # Mapper 层
│   │   │   ├── converter/              # MapStruct 对象转换器
│   │   │   ├── model/                  # 数据模型（entity/bo/dto/vo/form/query/event）
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
│       └── resources/                  # 测试配置（H2/TestContainers/SQL/模板）
```

## 三、核心模块

```mermaid
flowchart LR
    subgraph Security["安全认证"]
        SessionFilter["SessionFilter"]
        SecurityConfig["Spring Security"]
        PermissionService["权限校验"]
    end

    subgraph Storage["文件管理"]
        StorageInterface["StorageService 接口"]
        Minio["MinioFileService"]
        Local["LocalFileService"]
        NginxStatic["NginxStaticFileService"]
        Factory["FileBOFactory"]
    end

    subgraph System["系统管理"]
        RBAC["RBAC 权限模型"]
        DeptTree["部门树形结构"]
    end

    subgraph Algorithm["算法管理"]
        AlgoCtrl["SysAlgorithmController"]
        PythonClient["Python 服务 HTTP 客户端"]
    end

    subgraph CoreBusiness["核心业务模块"]
        Dehaze["去雾处理<br/>SysInputHistoryService + prediction 拦截器链"]
        Compare["效果对比<br/>CompareService"]
        AlgoSelect["算法选择<br/>AlgorithmSelectService"]
        Recommend["推荐管理<br/>RecommendationService"]
        Favorite["收藏管理<br/>FavoriteService"]
    end

    subgraph ImportExport["通用导入导出"]
        GenericCtrl["GenericImportExportController"]
        ExportRegistry["ExportHandlerRegistry"]
        ImportRegistry["ImportHandlerRegistry"]
        Strategy["GenericExportStrategy / GenericImportStrategy"]
    end

    SessionFilter --> SecurityConfig
    SecurityConfig --> PermissionService
    Factory --> Minio
    Factory --> Local
    Factory --> NginxStatic
    AlgoCtrl --> PythonClient
    Dehaze --> AlgoCtrl
    Compare --> AlgoCtrl
    AlgoSelect --> AlgoCtrl
    AlgoSelect --> Recommend
    AlgoSelect --> Favorite
    GenericCtrl --> ExportRegistry
    GenericCtrl --> ImportRegistry
    ExportRegistry --> Strategy
    ImportRegistry --> Strategy
```

### 3.1 安全认证模块

| 组件 | 实现 | 说明 |
|------|------|------|
| Session 认证 | Redis `session:{sessionId}` | 存储 userId、username、authorities，TTL 7 天，剩余 < 24h 自动续期 |
| 密码加密 | BCryptPasswordEncoder | Spring Security 标准实现 |
| 验证码 | Hutool Captcha | 支持圆圈/GIF/干扰线/扭曲多种类型 |
| UserDetails | SysUserDetailsService | 从数据库加载用户信息 |

RBAC 权限模型：

```
用户 -> 角色（多对多） -> 权限标识（多对多）
权限格式: 模块:功能:操作（如 sys:user:add）
```

三层安全防护：SessionFilter 会话校验 -> Redis 权限校验 -> 方法级 @PreAuthorize + @DataPermission 注解。

权限缓存以逐角色独立 Key 存储：`role:perms:{roleCode}`，值为纯 JSON 字符串数组，使用 StringRedisTemplate 读写，三端（Java/Go/Python）格式统一。

### 3.2 文件管理模块

策略模式适配多存储后端（minio/local/nginx-static）：

- `sys_file` 表只存 `object_name` + `storage`（与环境无关）
- URL 运行时拼接不落库（`storage.baseUrl + object_name`）
- 下载按 `storage` 选后端读取，无前缀判断分支
- 环境迁移只改配置不改库

### 3.3 通用导入导出模块

Handler 模式 + 通用策略实现，各业务模块只需实现 ExportHandler/ImportHandler 接口：

| 模块 | ExportHandler | ImportHandler |
|------|--------------|--------------|
| 用户管理 | UserExportHandler | UserImportHandler |
| 角色管理 | RoleExportHandler | RoleImportHandler |
| 部门管理 | DeptExportHandler | DeptImportHandler |
| 菜单管理 | MenuExportHandler | MenuImportHandler |
| 字典管理 | DictExportHandler | DictImportHandler |
| 数据集管理 | DatasetExportHandler | -（仅导出） |
| 算法管理 | AlgorithmExportHandler | AlgorithmImportHandler |

### 3.4 插件化扩展组件

通过 `plugin/` 包实现可插拔的横切关注点：

| 插件 | 注解 | 实现方式 |
|------|------|----------|
| 防重复提交 | `@PreventDuplicateSubmit` | AOP + Redisson 分布式锁 |
| 接口限流 | `@RateLimit` | AOP + Redisson 令牌桶/固定窗口 |
| 数据权限 | `@DataPermission` | MyBatis-Plus 拦截器 |
| 字段自动填充 | `@TableField(fill=...)` | MetaObjectHandler |

### 3.5 去雾处理模块

预测主流程通过 `prediction/` 包的拦截器链（`PredictionInterceptor`）实现可插拔扩展：拦截器命中则短路不调用 Python 算法服务，未命中则继续主流程委托算法管理模块执行推理。预测日志、输入历史、参数预设分别由 `SysPredLogService`、`SysInputHistoryService`、`SysPresetService` 承担。

异步任务状态（处理中/已完成/已失败）与任务管理模块语义对齐，但物理存储于 `sys_input_history` 表而非 `sys_task` 表——这是为保留用户维度的输入历史视图而做的存储分叉，状态查询走 `SysInputHistoryService` 而非统一任务接口。

VIP 配额校验在预测请求入口执行：处理前预校验、成功后实扣减、失败不扣减，Redis 原子操作防止并发超扣。组件实现详见 [去雾处理/后端实现.md](../../03-模块设计/核心模块/去雾处理/后端实现.md)。

### 3.6 效果对比模块

多模式对比（并排/重叠/放大镜/指标）与评估指标计算（PSNR/SSIM/LPIPS/NIQE/Entropy）由 `CompareService` 统一编排，指标计算委托算法管理模块调用 Python 服务完成。对比报告异步生成复用任务管理模块框架，报告文件存入 MinIO 保留 24 小时。组件实现详见 [效果对比/后端实现.md](../../03-模块设计/核心模块/效果对比/后端实现.md)。

### 3.7 算法选择模块

`AlgorithmSelectService` 组合搜索、推荐、收藏状态构建前端算法视图，委托算法管理模块完成算法检索。实验性算法通过会员管理模块校验 VIP 可见性。组件实现详见 [算法选择/后端实现.md](../../03-模块设计/核心模块/算法选择/后端实现.md)。

### 3.8 收藏管理模块

统一收藏表 `sys_favorite` 通过 `target_type` 区分收藏对象类型（algorithm/result/dataset），新模块接入收藏只需声明 targetType，无需重复开发表/接口。`FavoriteService` 提供添加/取消/列表/状态批量查询/计数，收藏状态批量查询接口供各业务模块在加载列表时标记每条记录收藏状态。VIP 收藏容量校验在收藏操作前执行（普通用户 200 条、VIP 用户 500 条）。组件实现详见 [收藏管理/后端实现.md](../../03-模块设计/基础模块/收藏管理/后端实现.md)。

### 3.9 推荐管理模块

`RecommendationService` 编排推荐流程：图像特征提取 → 规则匹配 → 排序 → 结果构建。当前采用规则匹配引擎（可解释性强、管理员可配置场景→算法映射），冷启动策略为新算法赋予默认评分并随机曝光。组件实现详见 [推荐管理/后端实现.md](../../03-模块设计/基础模块/推荐管理/后端实现.md)。

## 四、缓存体系

```mermaid
flowchart TB
    subgraph CacheArch["缓存体系"]
        subgraph SpringCache["Spring Cache (注解式)"]
            Annotation["@Cacheable / @CacheEvict"]
        end

        subgraph MultiLevel["多级缓存"]
            L1["L1 Caffeine 本地缓存 (5min TTL)"]
            L2["L2 Redis 分布式缓存 (1h TTL)"]
        end

        subgraph Redisson["Redisson"]
            Lock["分布式锁"]
            RateLimit["限流器"]
        end
    end

    SpringCache --> MultiLevel
    L1 --> L2
```

多级缓存后端类型为 Caffeine L1 + Redis L2，通过 MultiLevelCacheManager 管理。Spring Cache 注解使用覆盖 menu、dataset、role 等模块，共 10 处。

## 五、消息队列

异步任务通过 RabbitMQ 解耦：业务侧创建任务记录落库后，由 `taskExecutor.publishExportTask()` 发布消息，消费者调用 `taskExecutor.executeExportTask()` 执行。组件加载受 `@ConditionalOnProperty(rabbitmq.enabled)` 控制，与 Go/Python 端共享同一 Exchange/Queue 拓扑。

```mermaid
flowchart LR
    subgraph Producer["生产端"]
        TaskService --> Publisher["RabbitMQPublisher"]
    end

    subgraph Broker["RabbitMQ"]
        Exchange["dehaze.tasks (direct)"]
        Export["业务队列<br/>TTL 24h"]
        Retry0["retry.0<br/>TTL 5s"]
        Retry1["retry.1<br/>TTL 30s"]
        Retry2["retry.2<br/>TTL 5min"]
        DLX["*.dlx 死信队列"]
    end

    subgraph Consumer["消费端"]
        ExportConsumer["业务消费者"]
        DlxConsumer["死信消费者"]
    end

    Publisher --> Exchange
    Exchange --> Export
    Export -.nack.-> Retry0
    Retry0 -.超时.-> Retry1
    Retry1 -.超时.-> Retry2
    Retry2 -.超时.-> DLX
    Export --> ExportConsumer
    DLX --> DlxConsumer
```

| 业务队列 | 用途 | 消费者 |
|---------|------|--------|
| `task.export` | 导出任务（数据集/用户/角色/部门/菜单/字典/算法） | ExportTaskConsumer |
| `feedback.low_rating` | 低分评价告警 | LowRatingAlertConsumer |

每条业务队列配套 3 级重试队列（`retry.0` 5s → `retry.1` 30s → `retry.2` 5min），通过 DLX 实现阶梯重试，最终进入 `*.dlx` 死信队列由 DlxConsumer 兜底处理。

可靠性机制：

| 机制 | 实现 |
|------|------|
| 消费确认 | 手动 ACK（`MANUAL`），`defaultRequeueRejected=false`：消费失败不入原队列，转入重试阶梯，避免毒消息阻塞 |
| 发送确认 | Publisher Confirm + Return 回调，发送失败/不可路由时记录日志 |
| 消费幂等 | 基于任务终态校验（`TERMINAL_STATUSES`），终态任务跳过重复消费 |
| 并发控制 | 消费者并发 3-10，prefetch 10 |

## 六、安全过滤器链

```mermaid
flowchart LR
    Req["请求"] --> Trace["TraceIdFilter<br/>TraceID 生成/透传/回写 MDC"]
    Trace --> CORS["CorsFilter (order=-101)"]
    CORS --> Log["RequestLogFilter<br/>请求访问日志"]
    Log --> ApiKey["ApiKeyAuthenticationFilter<br/>API Key 认证"]
    ApiKey --> Session["SessionFilter<br/>Session 校验"]
    Session --> Security["Spring Security FilterChain"]
    Security --> Permission["@PreAuthorize 权限校验"]
    Permission --> Handler["业务处理 Controller"]
```

| 过滤器 | 功能 | 作用范围 |
|--------|------|----------|
| TraceIdFilter | TraceID 生成/透传/回写 MDC（`@Order(HIGHEST_PRECEDENCE)`，最先执行） | 全局 |
| CorsFilter | 跨域资源共享 | 全局 |
| RequestLogFilter | 每请求一条访问日志（status/duration） | 全局 |
| ApiKeyAuthenticationFilter | `dhak_*` 形式 API Key 认证，与 Session 认证解耦，优先于 SessionFilter | 受保护路由 |
| SessionFilter | Session 验证、SecurityContext 注入 | 受保护路由 |
| SecurityFilterChain | Spring Security 认证/授权链 | 全局 |

异步线程（`@Async`）通过 `AsyncConfig` 的 TaskDecorator 透传 MDC（traceId/method/path/ip/userId）与 SecurityContext，保证异步方法日志链路追踪和权限上下文不中断。

安全工具：XssUtils（XSS 过滤）、PathSecurityUtil（路径穿越检测）、SecurityUtils（获取当前用户上下文）。

## 七、数据访问层

| 组件 | 选型 | 说明 |
|------|------|------|
| ORM | MyBatis-Plus 3.5.5 | 通用 CRUD、分页、数据权限 |
| 连接池 | Druid 1.2.16 | 监控、防 SQL 注入、连接管理 |
| 关系数据库 | MySQL | 业务数据（生产环境） |
| 文档数据库 | MongoDB | 登录日志（LoginLog）、审计日志（AuditLog），启动时由 MongoConfig 自动建索引 |
| 测试数据库 | H2 / TestContainers(MySQL) | 单元测试 / 集成测试 |

MyBatis-Plus 插件链：

```mermaid
flowchart LR
    SQL["SQL 执行"] --> DP["DataPermissionInterceptor 数据权限拦截"]
    DP --> Page["PaginationInnerInterceptor 分页插件"]
    Page --> DB[("MySQL")]
```

数据权限（DataScope）：基于 MyBatis-Plus DataPermissionHandler 实现行级数据权限控制，支持全部数据/本部门/本部门及下级/仅本人四种范围。

自动填充：INSERT/UPDATE 时自动填充 `createTime`/`updateTime`/`createBy`/`updateBy`。

逻辑删除：全局配置 `deleted` 字段（0=未删除，1=已删除）。

## 八、定时任务

统一采用 XXL-Job 分布式调度（调度周期由 XXL-Job Admin 统一管理，与 Go/Python 端共享调度配置），`@Scheduled` 未使用。所有 Job 通过 `@XxlJob` 注解声明 handler，执行前注入 SystemSecurityContext 以系统身份运行，避免无用户上下文导致的权限校验失败。

| 业务域 | Handler | 功能 |
|--------|---------|------|
| 任务管理 | `cleanupExpiredTasks` | 每天 02:00 物理删除 7 天前已完成/取消任务、30 天前已终止任务 |
| 任务管理 | `cleanupStuckTasks` | 每小时将 PROCESSING 超 30min、PENDING 超 24h 的僵死任务标记为失败并清除缓存 |
| 任务管理 | `cleanupStuckPredEvalLogs` | 预测/评估日志过期清理 |
| 订单 | `expireOrders` | 待支付订单超 30 分钟自动取消 |
| 订单 | `completeExpiredOrders` | 过期订单自动完成 |
| 订单 | `retryFailedRefunds` | 退款失败记录重试 |
| 会员 | `resetMonthlyQuota` | 每月 1 日重置 VIP 月度配额 |
| 会员 | `sendExpireReminders` | 会员到期前提醒 |
| 会员 | `processExpiredMembers` | 会员过期状态处理 |
| 营销 | `expireUserCoupons` | 用户优惠券过期失效 |
| 营销 | `autoRenew` | 自动续费扣款 |
| 消息 | `cleanupExpiredMessages` | 过期消息清理 |
| 消息 | `refreshUnreadCountCache` | 未读数缓存刷新 |
| 消息 | `processDelayedPush` | 延迟消息推送 |
| 公告 | `sendScheduledAnnouncements` | 定时公告发送 |

## 九、配置管理

多环境支持：

| 环境 | Profile | 特性差异 |
|------|---------|----------|
| 开发 | dev | 慢SQL日志(>1s)、Swagger 启用、DevTools 热重载 |
| 测试 | test | H2 内存数据库 / TestContainers、缓存禁用 |
| 生产 | prod | Swagger 禁用、连接池优化、日志输出到文件 |

敏感信息通过环境变量注入，YAML 中使用 `${ENV_VAR}` 占位符。条件化装配通过 `@ConditionalOnProperty` 控制 XXL-Job、Redis Cache 等组件按需加载。

## 十、统一响应与错误处理

响应字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| code | string | 5 位错误码，`00000` 表示成功 |
| msg | string | 提示信息 |
| data | object | 业务数据 |
| traceId | string | 链路追踪 ID，取自 MDC |
| timestamp | long | 服务器时间戳 |
| errors | array | 字段校验错误明细 |

错误码采用 5 位字符串编码，与 Go/Python 端保持一致：

| 前缀 | 类别 | 示例 |
|------|------|------|
| `00` | 成功 | `00000` - 一切 ok |
| `A0` | 用户端错误 | `A0001` - 用户端错误, `A0200` - 登录异常, `A0400` - 参数错误 |
| `B0` | 系统执行错误 | `B0001` - 系统执行出错, `B0210` - 并发限流 |
| `C0` | 第三方服务错误 | `C0001` - 调用第三方服务出错, `C0300` - 数据库服务出错 |

全局异常处理通过 `@RestControllerAdvice` + `@ExceptionHandler` 统一拦截并格式化输出，共处理 18 类异常。

## 十一、应用生命周期

### 启动流程

```mermaid
sequenceDiagram
    participant Main as SystemApplication
    participant Boot as SpringBoot
    participant Bean as Bean 初始化
    participant Server as 内嵌 Tomcat

    Main->>Boot: SpringApplication.run()
    Boot->>Boot: 加载 application.yml (Profile 切换/环境变量展开)
    Boot->>Bean: @Configuration 扫描 (Security/Mybatis/Redis/Cache等)
    Bean->>Server: 启动内嵌 Tomcat
    Server->>Server: 注册 Filter 链
```

### 优雅关闭

收到 SIGINT/SIGTERM -> Tomcat 停止接收新连接 -> 等待 in-flight 请求完成（默认 30s 超时） -> 销毁 Spring Bean -> 关闭数据源/Redis/线程池连接池。

本地开发统一通过项目根目录 `scripts/run.py` 管理三端后端的生命周期。

## 十二、三端对照

| 基础设施能力 | dehaze-java | dehaze-go | 一致性 |
|-------------|-------------|-----------|--------|
| HTTP 框架 | Spring MVC (Tomcat) | Gin | 接口语义一致 |
| ORM | MyBatis-Plus | GORM | 功能对等 |
| 缓存 | Spring Cache + Caffeine L1 + Redis L2 | 多级缓存 (gokit local + Redis) | 已对齐多级缓存 |
| 分布式锁 | Redisson | go-redis | 语义一致 |
| 消息队列 | RabbitMQ | RabbitMQ | 共享 Exchange/Queue |
| 定时任务 | @Scheduled + XXL-Job | Ticker + XXL-Job | 共享 XXL-Job Admin |
| 日志 | Logback | Zap | 格式/级别统一 |
| 认证 | Spring Security + Session | 自研中间件 + Session | Session ID 互通 |
| 权限 | RBAC (@PreAuthorize) | RBAC (中间件) | 权限标识一致 |
| 数据权限 | MyBatis-Plus 拦截器 | GORM Plugin | 语义一致 |
| 错误码 | 5 位字符串 (A0/B0/C0) | 5 位字符串 (A0/B0/C0) | 完全一致 |
| 响应格式 | `{code, msg, data}` | `{code, msg, data}` | 完全一致 |
| TraceID | TraceIdFilter + MDC | trace.go + Context | 语义一致 |

## 十三、关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 框架 | Spring Boot 3.3 | Java 生态标准，自动配置 + 起步依赖 |
| 安全 | Spring Security + Redis Session | RBAC 细粒度权限控制 |
| ORM | MyBatis-Plus | 通用 CRUD、分页、数据权限插件 |
| 缓存 | Caffeine L1 + Redis L2 多级缓存 | 降低 Redis 压力，提升响应速度 |
| 消息队列 | RabbitMQ | 与 Go/Python 端统一中间件 |
| 文件存储 | 策略模式适配多后端 | minio/local/nginx-static 统一抽象 |
| 导入导出 | Handler 模式 + 通用策略 | 各模块只需实现接口，复用框架 |
| 对象转换 | MapStruct | 编译期生成，避免运行时反射开销 |
| 定时任务 | XXL-Job | 分布式调度、Web 管理控制台，与 Go/Python 端共享调度配置 |
| 日志 | SLF4J + Logback | 详见 [日志架构设计](../../02-系统架构/07-日志架构设计.md) |
| 监控 | Micrometer + Prometheus | 指标采集，与 Go 端命名统一 |
| 收藏统一抽象 | `sys_favorite` 表 + `target_type` 区分 | 新模块接入收藏只需声明 targetType，无需重复开发表/接口/组件 |
| 推荐引擎选型 | 规则匹配引擎 | 规则可解释性强、可快速上线、管理员可视化配置场景→算法映射 |
| VIP 配额校验 | 拦截器模式 + Redis 原子扣减 | 处理前预校验、处理成功后实扣减、失败不扣减，保证配额与处理结果一致性；Redis 原子操作（DECR + 阈值判断）防止并发超扣 |

## 十四、可观测性

| 维度 | 实现 |
|------|------|
| 指标采集 | Micrometer + Prometheus，业务计数器由各 Service 通过 MeterRegistry 递增：`dehaze_prediction_total`、`dehaze_evaluation_total`、`dehaze_task_total`、`dehaze_file_upload_total`；Python 调用耗时 `dehaze_python_call_duration` |
| 链路追踪 | TraceIdFilter 生成 TraceID 写入 MDC，异步线程通过 TaskDecorator 透传（详见第六节） |
| 日志 | Logback 结构化日志（traceId/method/path/status/duration），详见 [日志架构设计](../../02-系统架构/07-日志架构设计.md) |
| 访问日志 | RequestLogFilter 每请求一条 INFO ACCESS 日志 |
