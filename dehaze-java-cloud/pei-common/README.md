### `pei-common` 包分析

`pei-common` 是一个典型的 **通用工具库模块**，其设计目的是为整个项目提供可复用的公共组件、基础类库和通用功能。它被划分为多个子包，每个子包专注于某一特定功能领域。

---

## 一、为什么单独构建 `pei-common` 及其子包？

### 1. **代码复用性**

- 将常用功能封装为独立模块，避免重复开发。
- 多个业务模块（如 `ruoyi-system`, `ruoyi-resource`）可以统一引用这些公共组件。

### 2. **模块解耦**

- 每个子包职责单一，降低模块间耦合度。
- 如 `pei-common-redis` 只负责 Redis 相关操作，`pei-common-sms` 只处理短信发送逻辑。

### 3. **版本管理与依赖控制**

- 使用 `pei-common-bom` 统一管理所有子包的版本。
- 避免不同模块引入不一致版本导致的问题。

### 4. **便于维护与扩展**

- 当需要升级某个通用功能时，只需修改对应的子包，不影响其他模块。
- 新增功能只需新增子包，不影响现有结构。

---

## 二、各主要子包作用详解

| 子包名称                         | 作用                                                                                                         | 设计原因                  |
|------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------|
| **pei-common-bom**           | 提供所有 [pei-common-*](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-bom\pom.xml) 子包的依赖版本管理 | 统一版本，避免冲突             |
| **pei-common-core**          | 核心工具类（如字符串处理、日期格式化、常量定义等）                                                                                  | 基础支持，几乎所有模块都会依赖       |
| **pei-common-dict**          | 字典数据管理                                                                                                     | 统一系统中使用的静态数据（如性别、状态码） |
| **pei-common-dubbo**         | Dubbo 服务调用支持                                                                                               | 微服务间通信标准化             |
| **pei-common-elasticsearch** | Elasticsearch 操作封装                                                                                         | 支持全文检索功能              |
| **pei-common-encrypt**       | 加密解密工具类（如 AES、MD5、SHA）                                                                                     | 数据安全处理                |
| **pei-common-excel**         | Excel 导入导出支持（基于 Apache POI）                                                                                | 数据导入导出标准化             |
| **pei-common-idempotent**    | 幂等性校验工具                                                                                                    | 防止重复提交                |
| **pei-common-job**           | 定时任务调度支持                                                                                                   | 统一任务调度机制              |
| **pei-common-json**          | JSON 序列化/反序列化工具（如 Jackson、FastJSON）                                                                        | 数据交换标准格式              |
| **pei-common-loadbalancer**  | 负载均衡策略配置                                                                                                   | 微服务调用时负载均衡支持          |
| **pei-common-log**           | 日志记录工具封装（如 AOP 记录请求日志）                                                                                     | 统一日志输出格式              |
| **pei-common-mail**          | 邮件发送支持                                                                                                     | 系统通知、告警等功能            |
| **pei-common-mybatis**       | MyBatis 扩展支持（如自动填充、乐观锁）                                                                                    | ORM 操作增强              |
| **pei-common-nacos**         | Nacos 配置中心集成                                                                                               | 动态配置管理                |
| **pei-common-oss**           | 对象存储服务封装（如阿里云 OSS、MinIO）                                                                                   | 文件上传下载标准化             |
| **pei-common-prometheus**    | Prometheus 监控指标暴露                                                                                          | 系统运行状态监控              |
| **pei-common-ratelimiter**   | 限流工具（如 Guava RateLimiter）                                                                                  | 控制接口访问频率              |
| **pei-common-redis**         | Redis 缓存操作封装                                                                                               | 提升系统响应速度              |
| **pei-common-satoken**       | Sa-Token 权限框架集成                                                                                            | 统一认证授权机制              |
| **pei-common-seata**         | Seata 分布式事务集成                                                                                              | 保证跨服务事务一致性            |
| **pei-common-security**      | Spring Security 集成                                                                                         | 权限控制基础支持              |
| **pei-common-sensitive**     | 敏感词过滤工具                                                                                                    | 内容安全控制                |
| **pei-common-sentinel**      | Alibaba Sentinel 流控组件集成                                                                                    | 服务熔断降级支持              |
| **pei-common-skylog**        | SkyWalking 日志集成                                                                                            | 分布式链路追踪支持             |
| **pei-common-sms**           | 短信服务封装（如阿里云 SMS、腾讯云 SMS）                                                                                   | 通知、验证码发送              |
| **pei-common-social**        | 社交登录支持（如微信、QQ、GitHub 登录）                                                                                   | 第三方登录集成               |
| **pei-common-sse**           | Server-Sent Events 支持                                                                                      | 实时消息推送                |
| **pei-common-tenant**        | 多租户支持                                                                                                      | SaaS 架构下隔离不同租户数据      |
| **pei-common-translation**   | 国际化多语言支持                                                                                                   | 多语言切换                 |
| **pei-common-web**           | Web 层通用支持（如 BaseController、ResponseEntity）                                                                 | 统一返回格式、异常处理           |
| **pei-common-websocket**     | WebSocket 支持                                                                                               | 实时通信                  |

---

## 三、最佳实践建议

| 类型        | 最佳实践                      |
|-----------|---------------------------|
| **模块划分**  | 按功能细粒度拆分，保持每个子包职责单一。      |
| **版本管理**  | 使用 BOM 统一管理依赖版本，确保一致性。    |
| **日志记录**  | 所有公共组件应支持日志输出，便于调试。       |
| **异常处理**  | 自定义统一异常类型，避免底层错误暴露。       |
| **缓存机制**  | 对高频读取操作增加本地或分布式缓存。        |
| **幂等性控制** | 对写操作接口增加 Token 或唯一 ID 校验。 |
| **测试覆盖**  | 所有核心工具类应有单元测试，确保稳定性。      |

---

## ✅ 总结

`pei-common` 及其子包的设计体现了良好的模块化思想和面向接口编程理念。通过将通用功能抽象为独立模块，不仅提升了系统的可维护性和可扩展性，也为后续新功能的接入提供了统一的标准和规范。

这种设计非常适合企业级微服务架构，能够有效支撑复杂系统的长期演进与迭代。

## 📦 `pei-common-core` 模块概述

`pei-common-core` 是 RuoYi-Cloud-Plus
项目中的核心公共库，提供通用的功能支持。它封装了系统中各个模块共用的基础能力，如统一响应结构、异常处理、工具类、常量定义、枚举类等，确保整个项目的代码风格一致性和可维护性。

---

## 🧩 主要功能及其实现

### 1. **统一响应结构**

- **作用**：为所有接口返回统一格式的数据结构，便于前端解析和错误处理。
- **实现方式**：
  -
  在 [R.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java)
  中定义泛型类 `R<T>`，包含状态码、消息内容和数据对象。
- **关键方法**：
    - [ok(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java#L46-L48)
      ：成功响应。
    - [fail(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java#L66-L68)
      ：失败响应。
    - [warn(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java#L88-L90)
      ：警告响应。
    - [isSuccess(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java#L115-L117) / [isError(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\R.java#L111-L113)
      ：用于判断响应是否成功。

---

### 2. **业务状态枚举（BusinessStatusEnum）**

- **作用**：定义常见的业务状态码及其描述，用于流程控制和状态判断。
- **实现方式**：
    - 使用 Java 枚举类封装状态码和描述信息。
    - 提供静态方法进行状态匹配、校验和查询。
- **关键方法**：
    - [getByStatus(String status)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\enums\BusinessStatusEnum.java#L77-L80)
      ：根据状态码获取对应的枚举。
    - [findByStatus(String status)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\enums\BusinessStatusEnum.java#L88-L94)
      ：获取状态码对应的描述。
    - [checkStartStatus(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\enums\BusinessStatusEnum.java#L141-L153) / [checkCancelStatus(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\enums\BusinessStatusEnum.java#L160-L174)
      ：用于流程状态校验，抛出 [ServiceException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\ServiceException.java#L14-L69)。
- **典型使用场景**：
    - 流程审批状态管理。
    - 单据状态变更时的合法性校验。

---

### 3. **字符串工具类（StringUtils）**

- **作用**：提供丰富的字符串操作方法，增强开发效率。
- **实现方式**：
    - 继承 `org.apache.commons.lang3.StringUtils` 并扩展更多实用方法。
    - 集成 Hutool 和 Spring 的 `AntPathMatcher` 工具。
- **关键方法**：
    - [format(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L100-L102)
      ：格式化字符串，支持占位符替换。
    - [str2List(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L134-L156) / [str2Set(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L121-L123)
      ：字符串与集合之间的转换。
    - [matches(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L211-L221) / [isMatch(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L232-L235)
      ：路径匹配（支持通配符）。
    - [padl(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StringUtils.java#L256-L270)
      ：数字或字符串左补零。
- **典型使用场景**：
    - 请求参数处理。
    - 路由匹配。
    - 日志输出格式化。

---

### 4. **异常处理体系**

- **作用**：统一处理系统级和服务级异常，提升系统的健壮性。
- **实现方式**：
  -
  定义基础异常类 [BaseException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\base\BaseException.java#L16-L72)。
  -
  扩展具体异常类如 [ServiceException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\ServiceException.java#L14-L69), [SseException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\SseException.java#L14-L61)
  等。
- **关键类**：
    - [ServiceException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\ServiceException.java#L14-L69)
      ：服务层通用异常。
    - [SseException](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\exception\SseException.java#L14-L61)
      ：Server-Sent Events 异常。
- **典型使用场景**：
    - 接口调用失败抛出异常。
    - 流程校验失败抛出业务异常。

---

### 5. **常量定义**

- **作用**：集中管理项目中常用的常量值，避免硬编码。
- **实现方式**：
  -
  多个常量类按功能划分，如 [Constants](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\Constants.java#L7-L74), [CacheConstants](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\CacheConstants.java#L7-L29), [HttpStatus](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\HttpStatus.java#L7-L92), [RegexConstants](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\RegexConstants.java#L11-L58)
  等。
- **典型常量类**：
    - [HttpStatus](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\HttpStatus.java#L7-L92)
      ：HTTP 状态码常量。
    - [CacheConstants](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\CacheConstants.java#L7-L29)
      ：缓存键名和过期时间。
    - [RegexConstants](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\constant\RegexConstants.java#L11-L58)
      ：正则表达式常量（如手机号、邮箱验证）。
- **典型使用场景**：
    - 响应码定义。
    - 缓存命名规范。
    - 表单字段校验规则。

---

### 6. **领域模型定义**

- **作用**：封装业务实体对象，作为数据传输载体。
- **实现方式**：
    - 在 `domain.model`
      包中定义 [LoginBody](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\model\LoginBody.java#L11-L42)
      等登录相关模型。
- **关键类**：
    - [LoginBody](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\domain\model\LoginBody.java#L11-L42)
      ：用户登录请求体。
- **典型使用场景**：
    - 接收用户登录参数。
    - 在认证过程中传递用户名、密码、验证码等信息。

---

### 7. **工具类集合**

- **作用**：提供常用工具方法，减少重复代码。
- **实现方式**：
    - 封装日期、流、对象、文件、IP、正则等常用工具类。
- **典型工具类**：
    - [DateUtils](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\DateUtils.java#L18-L286)
      ：日期格式化、计算时间差等。
    - [StreamUtils](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\StreamUtils.java#L18-L281)
      ：Java Stream 操作增强。
    - [ObjectUtils](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\ObjectUtils.java#L13-L59)
      ：空值检查、克隆等。
    - [ServletUtils](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\ServletUtils.java#L31-L288)
      ：Web 层辅助方法。
    - [SpringUtils](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-common\pei-common-core\src\main\java\com\pei\common\core\utils\SpringUtils.java#L14-L65)
      ：Spring 上下文获取工具。
- **典型使用场景**：
    - 时间戳处理。
    - 数据转换。
    - 文件上传下载。
    - IP 地址解析。

---

## 📁 包结构详解

```
com.pei.common.core
├── config/             // 配置类
├── constant/           // 常量定义
├── domain/             // 数据传输对象（DTO）
│   └── model/          // 领域模型
├── enums/              // 枚举类
├── exception/          // 异常类
├── factory/            // 工厂模式实现
├── service/            // 核心服务接口
├── utils/              // 各类工具类
├── validate/           // 参数校验工具
└── xss/                // XSS 过滤工具
```

---

## 🧠 技术栈与架构设计

### 技术栈

| 技术                  | 用途                      |
|---------------------|-------------------------|
| Lombok              | 减少样板代码（如 getter/setter） |
| Hutool              | 提供丰富工具方法                |
| Apache Commons Lang | 字符串处理                   |
| Spring Framework    | 获取上下文、Web 工具            |
| AntPathMatcher      | URL 路径匹配                |

### 架构图（文字描述）

```
[Controller] → [Service] → [Mapper]
      ↑            ↑            ↑
     DTO         Business       DAO
      ↓            ↓            ↓
   Response       Exception    Database
```

---

## ✅ 总结

`pei-common-core` 是整个微服务项目的核心依赖包，提供了以下核心能力：

- **统一响应结构**：通过 `R<T>` 类统一返回格式。
- **业务状态管理**：通过 `BusinessStatusEnum` 实现流程状态校验。
- **字符串操作增强**：提供丰富的字符串处理方法。
- **异常统一处理**：构建清晰的异常继承体系。
- **常量集中管理**：提高配置一致性。
- **工具类支持**：涵盖日常开发所需的各种工具。
