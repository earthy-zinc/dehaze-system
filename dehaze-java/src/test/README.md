# Dehaze Java 测试规范

本项目 Java 后端的测试规范与基础设施依赖处理策略，对标 `dehaze-python/tests/README.md`。
2026-08-23 决策：**H2 内存库与 Testcontainers 方案废弃**，数据库测试统一走真实 MySQL 测试库
（原因同 Python 端：H2/容器与 MySQL 存在 DDL 与 SQL 语义双层漂移，且容器方案要求本地 Docker，
与"隧道直连共享开发实例"的工作方式不符）。

## 1. 核心原则

1. **只替身"外部世界"，不 mock 内部逻辑**。数据库用真实 MySQL 测试库（dehaze_test）执行真实
   SQL；存储用 local 后端拼真实 URL；被测服务内部协作的 Service/Mapper 一律真实连接。
   只有真正的第三方外部依赖（Python 算法服务、支付渠道、外部 HTTP）才做 mock。
2. **断言行为而非实现**。断言"调用返回了什么结果、数据库状态发生了什么变化"，而不是
   "某个 mock 被调用了几次"。
3. **测试是资产不是补丁**。测试类与被测类同名对应（`FooService` → `FooServiceTest`/`FooServiceIT`），
   禁止 `FooServiceTestGaps`/`FooServiceRepro` 等补丁式文件堆积。
4. **中间件隔离铁律**：测试绝不写开发实例的业务数据——不连 Redis db 0、不连 Mongo `dehaze` 库、
   绝不消费共享 RabbitMQ 队列（详见 §4）。

## 2. 技术栈

| 依赖 | 用途 |
|------|------|
| JUnit 5 | 测试框架（surefire 跑 `*Test.java`，failsafe 跑 `*IT.java`） |
| Mockito | 纯单元测试 mock 协作者（`@ExtendWith(MockitoExtension.class)`） |
| AssertJ / JUnit Assertions | 断言 |
| spring-boot-test | `@SpringBootTest` 集成测试 |
| spring-security-test | Controller 层安全测试（`@WithMockUser` 等） |

## 3. 测试分层与命名

| 层次 | 命名 | 执行 | Spring 上下文 | 典型对象 |
|------|------|------|--------------|---------|
| 纯单元测试 | `*Test.java` | `mvn test`（surefire） | 不启动 | util、策略、注册表、handler（Mapper/Service 用 Mockito mock） |
| 集成测试 | `*IT.java` | `mvn verify`（failsafe） | `@SpringBootTest` 启动 | Service 落库行为、跨模块交互、并发/事务/级联 |
| 轻量切片 | `*Test.java` | `mvn test` | `@SpringBootTest(classes = TestConfig.class)` | 需要真实 Bean 装配但不需要 Redis/Security 自动配置的服务测试 |

- `TestConfig`（`config/TestConfig.java`）：排除 `SecurityAutoConfiguration`/`RedisAutoConfiguration`
  的精简上下文。**需要 `RedisTemplate` 或 `SecurityUtils` 生效的测试不要用它**，直接用默认
  `SystemApplication` 上下文（如 `TaskServiceIT`）。
- 测试目录镜像主包结构：`src/main/java/com/pei/dehaze/service/impl/` →
  `src/test/java/com/pei/dehaze/service/impl/`。

## 4. 基础设施依赖策略（核心）

**总开关原则**：RabbitMQ / XXL-Job / Kafka 由项目自有 `@ConditionalOnProperty` 开关装配
（`rabbitmq.enabled` / `xxl.job.enabled` / `kafka.enabled`），测试 profile 全部显式置
`false`——测试进程**不连、不消费、不注册**。需要测 MQ 消费逻辑时直接 new Consumer 调
handler 方法（Mockito mock `RabbitTemplate`/Mapper），不启监听容器。

| 依赖 | 纯单元测试 | 集成测试（*IT） | 关键约束 |
|------|------------|----------------|---------|
| **MySQL** | mock Mapper（仅当被测逻辑不涉 SQL 语义） | 真实 `dehaze_test` 库：`createDatabaseIfNotExist` 自动建库 → `sql.init` 加载 `schema/sys_*.sql`（DROP TABLE 幂等重建）+ `data/sys_*.sql` 种子 → `@Transactional` 测试结束自动回滚 | 通配必须限定 `sys_` 前缀：`xxl_job.sql` 含 `DROP DATABASE xxl_job`+`USE`，混入会切走连接当前库并误删共享实例的调度库 |
| **Redis** | Mockito mock `RedisTemplate`/`RedissonClient` | 真实实例（`${REDIS_HOST}:${REDIS_PORT}`，源为根 `.env`）**逻辑库 15**；`spring.cache.type: none` 禁用缓存抽象 | 禁止 db 0（开发后端在用）；测试写入的键必须带 TTL 或用后即删 |
| **MongoDB** | mock `MongoTemplate`/登录日志仓储 | 真实实例 **`dehaze_test` 库**（URI 与开发 `dehaze` 库隔离） | 登录日志/审计日志断言走行为断言（插入后可查回） |
| **RabbitMQ** | mock `RabbitTemplate`；Consumer 直接 new 后调方法 | **不触达**（`rabbitmq.enabled: false`） | 严禁测试进程消费共享队列——会把开发环境的导出任务 ack 掉 |
| **Kafka** | — | 不触达（`kafka.enabled: false`，Java 端未使用） | |
| **MinIO/OSS** | mock `FileService` | **local 后端**（`file.type: local`、`file.minio.enabled: false`、`file.local.upload-path: ./upload`） | 测试数据的 `sys_file.storage` 必须写 `"local"`——`StorageServiceFactory` 按启用的后端解析，写 `minio` 会抛"不支持的存储后端"；URL 断言用 `getUrl()` 的 `baseUrl + "/" + objectName` 拼接结果 |
| **XXL-Job** | 直接调用 handler 的 `@XxlJob` 方法 | **不触达**（`xxl.job.enabled: false`，不注册执行器/不开 netty 端口） | |
| **Elasticsearch** | — | — | **Java 端未使用**（ES 是 Python 端依赖，Python 侧用 respx mock） |
| **Python 算法服务** | Mockito mock HTTP 客户端 bean | 不调真实推理 | |
| **SecurityContext** | 归属校验逻辑通过构造/参数注入的可测设计优先 | 见 §5.2 模板：`SecurityContextHolder` + `SysUserDetails` | `SecurityUtils.getUserId()` 读 `SecurityContextHolder`，IT 不设上下文时为 null |

## 5. 测试模板

### 5.1 纯单元测试（无 Spring）

```java
@ExtendWith(MockitoExtension.class)
class ExportHandlerRegistryTest {
    @Test
    void testConstructor_DuplicateModuleThrows() {
        assertThrows(IllegalStateException.class,
                () -> new ExportHandlerRegistry(List.of(handler("user"), handler("user"))));
    }
}
```

### 5.2 数据库集成测试（真实 MySQL + 事务回滚 + 登录上下文）

被测方法经过 `SecurityUtils.getUserId()` 归属校验时，必须同时设置上下文与数据的
`create_by`（两者一致才通过校验）：

```java
@SpringBootTest                 // 需 Redis/Security 时用完整上下文；不需要时 classes = TestConfig.class
@Transactional                  // 测试结束自动回滚，种子数据零污染
class TaskServiceIT {
    private static final Long TEST_USER_ID = 1L;

    @BeforeEach
    void setUp() {
        SysUserDetails userDetails = new SysUserDetails();
        userDetails.setUserId(TEST_USER_ID);
        userDetails.setEnabled(true);
        userDetails.setAuthorities(Collections.emptySet());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities()));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();   // 泄漏会污染同 JVM 的其他测试
    }
    // 测试数据：task.setCreateBy(TEST_USER_ID);
}
```

并发/多线程场景不能用 `@Transactional` 回滚（子线程不在测试事务里），须在 `@AfterEach`
按测试专有标识（如随机 taskName 前缀）物理清理，参考 `DatasetConcurrencyIT`。

### 5.3 存储相关集成测试

测试数据 `sys_file.storage` 一律 `"local"`；断言 URL 包含 objectName 即可，不落盘
（`LocalFileService.getUrl` 是接口 default 实现，只拼 URL 不查文件）。

## 6. 运行方式

```bash
mvn test                     # 纯单元测试（surefire，*Test.java）
mvn verify                   # 单元 + 集成测试（failsafe，*IT.java）
mvn failsafe:integration-test -DskipTests=false   # 只跑 IT（需先 test-compile）
```

前置条件：
- 仓库根目录 `.env` 存在按基础设施分区的连接变量（MySQL/Redis/MongoDB 分别对应
  `MYSQL_HOST`/`MYSQL_PORT`、`REDIS_HOST`/`REDIS_PORT`、`MONGODB_HOST`/`MONGODB_PORT`，密码同理，
  远程实例需先 `ssh -L` 隧道：3306/6379/27017）
- MySQL/Redis/MongoDB 可达，否则 fail-fast（配置错误明确报错优于静默跳过）
- **与 dehaze-python 测试共用 `dehaze_test` 库**（Python conftest 会 DROP + CREATE 全量重置），
  两端数据库测试勿并行运行
- IntelliJ IDEA 首次运行需 `mvn process-test-resources` 触发 `config/sql` → classpath 复制，
  或启用 "Delegate IDE build/run actions to Maven"

## 7. 反模式（禁止）

1. H2 / embedded-redis / 内嵌中间件——方言漂移 + 生态库年久失修（2026-08-23 已废弃）
2. `@MockBean` 替换 Mapper/DataSource 把 IT 变成 mock 链——那应该写成纯单元测试
3. 测试连 Redis db 0 / Mongo `dehaze` 库——污染开发环境
4. 测试 profile 开启 `rabbitmq.enabled`——监听容器会真实消费共享队列
5. 测试数据 `storage="minio"` 而后端未启用——`StorageServiceFactory` 直接抛异常
6. `sql.init` 通配放开到 `*.sql`——`xxl_job.sql` 的 `USE xxl_job` 会切走连接当前库
7. 断言 mock 交互次数而非业务结果状态
8. 以 `xxxGaps`/`xxxRepro` 命名堆积补丁测试文件
