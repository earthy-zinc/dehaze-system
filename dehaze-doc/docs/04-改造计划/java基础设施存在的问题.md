# dehaze-java 后端特定问题分析与改进建议

> 文档定位：仅收录 **Java 特有问题**（性能/优化/bug），即由 Java 语言特性或 Spring 框架特性导致的实现缺陷。
>
> 跨三端的通用问题（架构设计层面 + 模块业务设计层面）已提取到 [通用基础设施问题与改进](./通用基础设施问题与改进.md)。
>
> 审查日期：2026-07-13
> 审查范围：`dehaze-java` 全部基础设施层代码
> 审查方法：对照 [09-后端基础设施设计(Java)](../02-系统架构/09-后端基础设施设计(Java).md) 核对实际实现

---

## 一、问题总览

| 领域 | 严重 | 高 | 中 | 低 |
|------|------|---|---|---|
| 安全认证 | 4 | 1 | 2 | 2 |
| 事务与数据一致性 | 4 | 0 | 1 | 0 |
| 异步任务与消息队列 | 1 | 1 | 1 | 0 |
| 缓存体系 | 0 | 1 | 1 | 1 |
| 外部服务客户端 | 1 | 2 | 1 | 0 |
| 可观测性 | 0 | 0 | 2 | 0 |
| 数据访问 | 0 | 0 | 1 | 1 |
| 文件存储 | 1 | 2 | 0 | 1 |

> 通用问题（TraceId、CORS、缓存防护、异步任务可靠性、监控端点鉴权、数据权限异步失效等）见 [通用基础设施问题与改进](./通用基础设施问题与改进.md)

---

## 二、安全认证（Java 特有）

### 2.1 [P0] JWT 不校验过期时间，过期 Token 永久有效

[JwtValidationFilter.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/filter/JwtValidationFilter.java#L58) 第 58 行仅调用 `JWTUtil.verify(token, secretKey)`，**只校验签名，不校验 `exp`/`nbf`/`iat`**。Hutool 5.8.x 的 `verify` 不含过期校验，需额外调用 `jwt.validate(leeway)`。

叠加"黑名单仅在注销时写入"（[AuthServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/AuthServiceImpl.java#L91) 第 91-115 行），结果是：**已过期但未主动注销的 Token 只要签名有效即可无限期访问**。

### 2.2 [P0] JWT 默认密钥为明文硬编码（含生产环境）

[application-dev.yml](file:///e:/DehazeSystem/dehaze-java/src/main/resources/application-dev.yml#L110) 第 110 行与 `application-prod.yml` 第 70 行：

```yaml
key: ${JWT_SECRET_KEY:SecretKey012345678901234567890123456789012345678901234567890123456789}
```

未设置环境变量时，**dev 与 prod 都回退到公开的硬编码密钥**，攻击者可伪造任意用户 JWT。

### 2.3 [P0] 登录无限流、无账户锁定

[AuthController.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/controller/AuthController.java#L24) 登录接口未标注 `@RateLimit`（项目已有 `plugin/ratelimit` 模块但未应用）。[SysUserDetails.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/model/SysUserDetails.java#L90) 第 90-92 行 `isAccountNonLocked()` 恒返回 `true`，`SysUser` 实体无 `failed_attempts`/`lock_until` 字段。

### 2.4 [P0] 无真正双 Token 机制，与架构文档不符

[AuthServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/AuthServiceImpl.java#L165) 第 165-184 行 `refreshToken()` 直接从当前仍有效的 Access Token 解析出 Authentication，重新签发一个新 Access Token。`SecurityProperties.JwtProperty` 只有 `key` 和 `ttl`，无独立 Refresh TTL。[JwtUtils.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/util/JwtUtils.java#L61) claims 中无 `typ` 区分 access/refresh。

> 对比：Go 端已实现真正的双 Token（`LoginTokenWithRefresh` + `RefreshToken`），Python 端也有 `JWT_ACCESS_TOKEN_EXPIRES` / `JWT_REFRESH_TOKEN_EXPIRES` 双 TTL。

### 2.5 [P1] 权限缓存部分命中不回源，权限被静默截断

[PermissionService.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/service/PermissionService.java#L84) 第 84-100 行：用户拥有角色 A、B，若 A 命中缓存、B 因部分驱逐而缺失，`perms` 非空则**不回源，B 的权限被静默丢弃**。回源后**不写回缓存**，每次部分缺失都重复查库。

### 2.6 [P1] 权限缓存全量刷新使用通配符——实际无效（Bug）

[SysRoleMenuServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysRoleMenuServiceImpl.java#L45) 第 45 行：

```java
redisTemplate.opsForHash().delete(SecurityConstants.ROLE_PERMS_PREFIX, "*");
```

`HDEL` 不支持 glob 通配，此处删除名为 `*` 的字段（几乎不存在），实际是 no-op。**已删除/改名的旧角色缓存不会被清理**，残留脏数据。

### 2.7 [中] 权限缓存无 TTL

[SysRoleMenuServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysRoleMenuServiceImpl.java#L52) 第 52、73、94 行 `opsForHash().put` 写入未设置过期时间，Hash 字段永不过期。

### 2.8 [中] XssUtils 黑名单正则可绕过，无全局过滤器

[XssUtils.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/common/util/XssUtils.java#L50) 第 50-65 行用正则移除 `<script>`/`<iframe>` 等标签——黑名单方式，可用 `<scr<script>ipt>`、`<svg onload>`、`<img onerror>` 等绕过。且 `filter/` 下无全局 XSS Servlet Filter，防护依赖业务层手动调用。

### 2.9 [低] SysUserDetailsService 无缓存、未实现账户锁定/过期

[SysUserDetailsService.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/service/SysUserDetailsService.java#L24) 每次登录直接查库，无用户信息缓存。`isAccountNonLocked`/`isAccountNonExpired`/`isCredentialsNonExpired` 均返回 `super`（恒 true）。

### 2.10 [低] JwtUtils 静态字段注入反模式 + 类型不一致

[JwtUtils.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/util/JwtUtils.java#L36) `private static byte[] key` + `@Value` setter 注入静态字段是反模式；`SecurityProperties.JwtProperty.ttl` 是 `Long` 而 `JwtUtils.ttl` 是 `int`，类型不一致。

---

## 三、事务与数据一致性（Java/Spring 特有）

### 3.1 [P0] 长事务包裹 MinIO 外部调用——连接池耗尽根因

`SysItemFileServiceImpl.saveItemFile`（第 72 行）本身未标注 `@Transactional`，但被以下外层 `@Transactional` 方法调用时，所有 MinIO 上传/删除全部加入外层事务：

| 外层 @Transactional 方法 | 行号 | 内部 MinIO 调用 |
|---|---|---|
| `DatasetOperationServiceImpl.createDatasetItemWithImages` | 第 70 行 | 循环 `saveItemFile`，N 张图 = 2N+1 次 MinIO 上传全在一个事务内 |
| `DatasetOperationServiceImpl.batchCreateDatasetItemsWithImages` | 第 166 行 | 外层循环多组，上传量成倍放大 |
| `DatasetOperationServiceImpl.deleteDatasetItemCascade` | 第 341 行 | 循环 `deleteFile`，每次 2 次 MinIO 删除 |
| `DatasetOperationServiceImpl.batchDeleteDatasetItemsCascadeWithResult` | 第 371 行 | 第 403-404 行循环 MinIO 删除 |
| `DatasetOperationServiceImpl.batchDeleteDatasets` | 第 432 行 | 第 471-473 行循环删除整棵树下所有 MinIO 对象 |
| `SysItemFileServiceImpl.deleteFile` | 第 136 行 | 第 149、154 行直接 `minioClient.removeObject` |

### 3.2 [P0] 自调用导致 @Transactional 失效

Spring 代理无法拦截同类内部方法调用：

1. [SysItemFileServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysItemFileServiceImpl.java#L83) 第 83 行 `saveItemFile` 调用 `this.saveItemFileRecord(...)`（第 112 行标注 `@Transactional`），自调用绕过代理，"短事务写入"设计被彻底架空
2. [SysItemFileServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysItemFileServiceImpl.java#L177) 第 177 行 `batchDelete` 调用 `this.deleteFile(id)`（第 136 行标注 `@Transactional`），自调用失效，批量删除无原子性
3. [DatasetOperationServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/DatasetOperationServiceImpl.java#L245) 第 245 行 `batchCreateDatasetItemsWithImages` 调用 `this.doCreateDatasetItemWithImages`，配合第 252 行 catch 吞异常，部分失败仍提交

### 3.3 [P0] @Transactional 内调用 @Async 引发竞态——任务读不到未提交数据

[TaskServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/TaskServiceImpl.java#L47) 第 47 行 `createTask`（`@Transactional`）在事务未提交时即调用 `taskExecutor.submitExportTask(sysTask.getId(), form)`（第 76 行, `@Async`）。[TaskExecutorImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/TaskExecutorImpl.java#L46) 第 51 行异步线程 `sysTaskMapper.selectById(taskId)` 大概率读不到该行（事务未提交），触发"任务不存在"分支直接 return。

### 3.4 [P0] try/catch 吞异常使 @Transactional 回滚语义失效

以下方法标注 `@Transactional(rollbackFor = Exception.class)`，但内部 try/catch 捕获并吞掉异常，事务永不回滚：

- `DatasetOperationServiceImpl.batchDeleteDatasetItemsCascadeWithResult`（第 371 行）：第 410 行 catch 后仅记录 failureDetail
- `DatasetOperationServiceImpl.batchDeleteDatasets`（第 432 行）：第 501、509 行 catch 后仅记录结果
- `DatasetOperationServiceImpl.batchCreateDatasetItemsWithImages`（第 166 行）：第 252 行 catch 后仅记录 failedItems

### 3.5 [中] @Transactional 缺少 rollbackFor，约定不一致

部分方法使用裸 `@Transactional`（默认仅对 RuntimeException 回滚），与项目 `@Transactional(rollbackFor = Exception.class)` 约定不一致：`SysAlgorithmVersionServiceImpl.addVersion`/`rollbackToVersion`、`SysUserServiceImpl.updateUser`、`SysRoleServiceImpl.assignMenusToRole`、`SysDictTypeServiceImpl.deleteDictTypes`。全代码库无 `propagation`/`isolation` 自定义，无法用 `REQUIRES_NEW` 切断长事务。

---

## 四、异步任务与消息队列（Java 特有）

### 4.1 [P0] TaskCleanupJob 完全失效——两个致命 Bug

[TaskCleanupJob.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/job/TaskCleanupJob.java)：

1. **Redis Key 前缀错误**（第 27 行）：`TASK_CACHE_PREFIX = "export:task:"`，而 `TaskConstants.TASK_CACHE_PREFIX = "task:"`
2. **状态字符串大小写不匹配**（第 48、112 行）：用小写 `"completed"`，而 `TaskConstants` 定义的是大写 `STATUS_COMPLETED = "COMPLETED"`，SQL 永远匹配不到任何行，**整个清理逻辑是死代码**

### 4.2 [P1] 消息队列全部禁用但代码存在——双路径混淆

`mq/` 目录下 9 个文件全部带 `@ConditionalOnProperty(... havingValue = "true")`，当前环境**完全不加载**（`rabbitmq.enabled: false`、`kafka.enabled: false`）。实际异步走 `@Async`。

- 三个任务消费者全是 stub，只有 TODO 注释
- `RabbitMQPublisher` 全代码库零注入
- 开发者可能误以为系统有 MQ 支撑，实际所有异步仅靠 4 线程的 `datasetTaskExecutor`

### 4.3 [中] 线程池配置不合理

[AsyncConfig.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/config/AsyncConfig.java#L25) `datasetTaskExecutor`：core=2、max=4、queue=10，`CallerRunsPolicy` 拒绝策略会让导出任务在 Tomcat 请求线程中同步执行（可能耗时数分钟），拖垮 Web 线程池。无 `AsyncUncaughtExceptionHandler`，异步异常静默丢失。

---

## 五、缓存体系（Java 特有）

### 5.1 [P1] Redis 反序列化存在 RCE 攻击面

[RedisConfig.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/config/RedisConfig.java#L42) 第 42-45 行与 [RedisCacheConfig.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/config/RedisCacheConfig.java#L64) 第 64-67 行启用 `activateDefaultTyping(..., NON_FINAL)` + `LaissezFaireSubTypeValidator`，允许反序列化任意非 final 类型，是已知的 Jackson 多态反序列化 RCE 攻击面。

### 5.2 [中] 缓存自调用导致 @Cacheable 失效

[SysDatasetServiceImpl.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysDatasetServiceImpl.java#L71) 第 71 行 `getAllDatasetStats()` 内部调用 `this.getAllDatasets()`（第 59 行标注 `@Cacheable`），同类自调用绕过 AOP 代理，缓存失效时直查 DB 拉全量数据集。

### 5.3 [低] RedisCacheConfig 注释与实际不符

[RedisCacheConfig.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/config/RedisCacheConfig.java#L29) 第 29 行注释写"xxl.job.enabled = true 才会自动装配"，实际检查 `spring.cache.enabled`。

---

## 六、外部服务客户端（Java 特有）

### 6.1 [P0] 自研熔断器失败率计算根本性错误

[PythonAlgorithmClient.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/client/PythonAlgorithmClient.java) 熔断器逻辑：

- **失败率计算错误**（第 181-203 行）：`recordFailure` 同时自增 `failureCount` 和 `totalCount`，但 `recordSuccess` 在 CLOSED 状态下**从不自增 `totalCount`**。导致 `failureRate = fails / fails = 100%`，**只要累计 10 次失败就必然触发熔断**
- **Per-endpoint 状态为死代码**（第 52-54 行）：`endpointStates`/`endpointFailureCounts`/`endpointTotalCounts` 声明后从未使用
- **HALF_OPEN 无并发控制**（第 168-179 行）：半开状态下任意线程均可发起调用
- **日志占位符错误**（第 197 行）：`"失败率 {:.1f}%"` 用了 printf 风格，SLF4J 占位符是 `{}`

### 6.2 [P1] RestTemplate 无连接池

[RestClientConfig.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/config/RestClientConfig.java#L33) 第 33-37 行默认 `RestTemplate` 走 JDK `HttpURLConnection`，**不维护连接池**。每次调用 Python 服务新建 TCP 连接，高并发下端口耗尽。

### 6.3 [中] 重试无幂等性保护

[PythonAlgorithmClient.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/client/PythonAlgorithmClient.java#L116) 第 116 行对 `ResourceAccessException`（网络超时）重试，但 `predict`/`evaluate` 均为 POST，若首请求已到 Python 端执行成功仅响应丢失，重试会触发重复处理，未发送幂等键。

### 6.4 [中] 预测/评估流 imageUrl 不可用

`SysPredLogServiceImpl.resolveImageUrl` 返回相对路径 `/api/v1/files/download/{fileId}`（带 TODO），Python 服务无法访问该 Java 内部路径，只有外部传入 `imageUrl` 的链路能工作。

---

## 七、可观测性（Java 特有）

### 7.1 [中] RequestLogFilter 形同虚设

[RequestLogFilter.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/filter/RequestLogFilter.java#L17) 继承 `CommonsRequestLoggingFilter` 但未调用 `setIncludeClientInfo(true)`/`setIncludeQueryString(true)`/`setIncludePayload(true)`，无响应状态、无耗时、无 traceId，生产排查基本无用。

### 7.2 [中] 业务指标完全缺失

`pom.xml` 引入了 `micrometer-registry-prometheus`，但全代码库**无一处** `Counter`/`Timer`/`Gauge`/`MeterRegistry` 使用。预测次数、评估耗时、Python 调用失败率、缓存命中率等业务指标全部缺失。

> 对比：Go 端有 Prometheus 中间件采集 HTTP 指标，Python 端有四大类指标（HTTP/GPU/推理/任务）。

---

## 八、数据访问（Java 特有）

### 8.1 [中] JDBC 启用 allowMultiQueries

[application-dev.yml](file:///e:/DehazeSystem/dehaze-java/src/main/resources/application-dev.yml#L29) 第 29 行 `allowMultiQueries=true` 允许堆叠查询，一旦任何位置出现 SQL 注入即可执行多语句，应移除。

### 8.2 [低] Mapper XML ${} 字符串插值

`SysRoleMenuMapper.xml`/`SysMenuMapper.xml` 用 `${@...Enum@getValue()}` 取静态枚举值，虽不可注入但绕过预编译缓存；`SysRoleMenuMapper.xml` 第 35 行 `type` 列未限定表别名（join 多表时歧义）。

---

## 九、文件存储（Java 特有）

### 9.1 [P0] MinIO 桶策略为 public 读写删——任意匿名客户端可篡改文件

[MinioFileService.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/file/MinioFileService.java#L205) 第 205-220 行 `publicBucketPolicy` 对 `Principal: {"AWS": ["*"]}` 开放了 `s3:GetObject`、`s3:PutObject`、`s3:DeleteObject`、`s3:ListBucket` 等操作。**任何能访问 MinIO endpoint 的客户端均可匿名读写删除文件**。

### 9.2 [P1] MinioFileService.downLoadFile 资源泄漏

[MinioFileService.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/file/MinioFileService.java#L160) 第 160-174 行：`GetObjectResponse` 继承 `InputStream`，持有底层 HTTP 连接，必须关闭。此处未用 try-with-resources，`response` 永不关闭，HTTP 连接泄漏。

### 9.3 [P1] FileBOFactory 临时文件泄漏

[FileBOFactory.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/common/util/FileBOFactory.java#L124) 第 124-125 行临时文件创建后赋给 `fileBO.setFile(tempFile)`，若后续 MinIO 上传失败或异常，临时文件**无人清理**，长期运行撑满临时目录。

### 9.4 [低] MinioFileService @PostConstruct 强依赖

第 66-73 行 `init()` 中调用 `createBucketIfAbsent`，MinIO 不可达时整个应用启动失败，无重试无降级。

---

## 十、优先级清单

### P0（阻断性）

| # | 问题 | 文件 |
|---|------|------|
| 1 | JWT 不校验过期 | JwtValidationFilter.java:58 |
| 2 | JWT 密钥硬编码 | application-dev.yml:110, application-prod.yml:70 |
| 3 | 登录无限流/无锁定 | AuthController.java, SysUser.java |
| 4 | 无真正双 Token | AuthServiceImpl.java:165, SecurityProperties.java |
| 5 | 长事务包裹 MinIO | DatasetOperationServiceImpl.java |
| 6 | 自调用 @Transactional 失效 | SysItemFileServiceImpl.java:83 |
| 7 | 事务内 @Async 竞态 | TaskServiceImpl.java:76 |
| 8 | try/catch 吞异常 | DatasetOperationServiceImpl.java:410,501,509,252 |
| 9 | TaskCleanupJob Bug | TaskCleanupJob.java:27,48,112 |
| 10 | 熔断器计算错误 | PythonAlgorithmClient.java:181-203 |
| 11 | MinIO 桶 public 读写删 | MinioFileService.java:205 |

### P1（重要）

| # | 问题 | 文件 |
|---|------|------|
| 12 | 权限缓存不回填 + 通配符失效 | PermissionService.java:84, SysRoleMenuServiceImpl.java:45 |
| 13 | MQ 死代码双路径 | mq/ 目录 |
| 14 | Redis 反序列化 RCE | RedisConfig.java:42, RedisCacheConfig.java:64 |
| 15 | RestTemplate 无连接池 | RestClientConfig.java:33 |
| 16 | MinioFileService 资源泄漏 | MinioFileService.java:160 |
| 17 | FileBOFactory 临时文件泄漏 | FileBOFactory.java:124 |
