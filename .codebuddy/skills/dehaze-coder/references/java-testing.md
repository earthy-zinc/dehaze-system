# Java 单元测试规范

适用于：`dehaze-java*` 系列后端项目。不适用于前端/JS/TS/Vitest、Go、Python 等任务。

## 测试优先级

1. **纯单元测试（默认）** —— 不启动 Spring 容器
    - 适用：业务逻辑/工具类/Service 纯逻辑
    - 工具：JUnit 5 + Mockito
    - 示例：`@Mock` / `@InjectMocks`，`when()` / `verify()`
2. **Web 层测试**
    - 适用：Controller 路由、参数校验、返回结构、异常映射
    - 示例：`@WebMvcTest` + MockMvc + `spring-security-test`
3. **集成测试** —— 需要验证真实装配/配置/拦截器/真实序列化/真实数据库时
    - 适用：集成验证，不要滥用
    - 示例：`@SpringBootTest`

## Mock 约束

必须 mock 的外部依赖（除非明确是对应依赖的集成测试）：

- 对象存储：MinIO/OSS 客户端
- 缓存：RedisTemplate/Redisson
- Mongo：MongoTemplate/Repository
- 调度：XXL-Job 触发接口
- 时间/随机数/UUID：必须可控（mock/注入 Clock/Random/UUID provider）

禁止：在测试里重新实现被测方法

## 用例结构

- 类结构（建议）：
    - `@DisplayName("...") class XxxTest { ... }`
    - 常量区、对象区、`@BeforeEach setUp()`、测试方法区
- 方法结构：
    - 准备数据与依赖
    - 调用被测方法
    - 断言关键结果

## 其他规则

- 私有字段不能直接访问，仅当别无选择时用反射辅助方法读取/设置（并限制在测试内使用）
- Security 场景：使用 `@WithMockUser` 或 `SecurityMockMvcRequestPostProcessors.user(...)`
- 复杂数据类型测试、多类型输入时，每种类型单独一个测试方法
- JSON：至少覆盖正确格式/错误格式/边界（空值、null）
- Setter 调用：Arrange 阶段避免链式调用，每行一个 setter（可读性优先）
