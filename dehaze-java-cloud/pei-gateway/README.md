
## 🌐 `pei-gateway` 模块概述

`pei-gateway` 是基于 Spring Cloud Gateway 构建的网关服务，主要负责请求的路由、鉴权、限流、日志记录等核心功能。它是整个微服务架构中的流量入口，所有外部请求都会经过该网关进行统一处理。

---

## 🧩 主要功能及其实现

### 1. **认证与鉴权**
- **作用**：拦截所有请求，验证用户是否已登录，是否有权限访问目标资源。
- **实现方式**：
    - 使用 [Sa-Token](https://github.com/dromara/sa-token) 进行权限控制。
    - 在 [AuthFilter.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\AuthFilter.java) 中注册全局过滤器，拦截 [/](file://E:\ProgramProject\RuoYi-Cloud-Plus\LICENSE) 下的所有请求。
    - 支持白名单配置（如 `/favicon.ico`, `/actuator` 等）。
- **关键方法**：
    - `setAuth(...)`：执行登录校验和客户端ID一致性检查。
    - `setError(...)`：处理未登录异常并返回统一格式错误信息。

### 2. **黑名单 URL 过滤**
- **作用**：阻止某些非法或恶意的URL访问。
- **实现方式**：
    - 在 [BlackListUrlFilter.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\BlackListUrlFilter.java) 中通过正则表达式匹配黑名单路径。
    - 配置文件中定义 [blacklistUrl](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\BlackListUrlFilter.java#L36-L36) 列表。
- **关键方法**：
    - [matchBlacklist(String url)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\BlackListUrlFilter.java#L40-L42)：判断当前请求URL是否匹配黑名单。

### 3. **全局日志记录**
- **作用**：记录每个请求的详细信息，包括请求路径、参数、耗时等。
- **实现方式**：
    - 在 [GlobalLogFilter.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\GlobalLogFilter.java) 中实现 `GlobalFilter` 接口。
    - 根据配置项 `spring.cloud.gateway.requestLog` 决定是否开启日志记录。
- **关键方法**：
    - [filter(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\filter\GlobalLogFilter.java#L37-L73)：在请求前后记录时间戳，并计算耗时。
    - [isJsonRequest(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\utils\WebFluxUtils.java#L51-L54)：判断是否为 JSON 请求体。

### 4. **自定义异常处理**
- **作用**：统一处理网关层抛出的异常，并返回结构化的错误响应。
- **实现方式**：
    - 在 [GatewayExceptionHandler.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\handler\GatewayExceptionHandler.java) 中实现 `ErrorWebExceptionHandler` 接口。
    - 处理常见异常类型（如 `NotFoundException`, `ResponseStatusException`）。
- **关键方法**：
    - [handle(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\handler\GatewayExceptionHandler.java#L23-L44)：根据异常类型构造不同的错误信息并写入响应。

### 5. **限流配置**
- **作用**：防止系统被高并发请求压垮，保障稳定性。
- **实现方式**：
    - 使用阿里 Sentinel 实现限流。
    - 在 [SentinelFallbackHandler.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\handler\SentinelFallbackHandler.java) 中处理限流降级逻辑。
- **配置类**：
    - [GatewayConfig.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\config\GatewayConfig.java) 注册了 Sentinel 的异常处理器。

### 6. **工具类支持**
- **作用**：提供 WebFlux 相关的工具方法。
- **实现方式**：
    - 在 [WebFluxUtils.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\utils\WebFluxUtils.java) 中封装了请求路径解析、Body读取、响应构建等功能。
- **关键方法**：
    - [getOriginalRequestUrl(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\utils\WebFluxUtils.java#L39-L44)：获取原始请求路径。
    - [resolveBodyFromCacheRequest(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\utils\WebFluxUtils.java#L83-L96)：从缓存中读取请求体内容。
    - [webFluxResponseWriter(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-gateway\src\main\java\com\pei\gateway\utils\WebFluxUtils.java#L105-L107)：统一响应格式输出。

---

## 📁 包结构详解

```
com.pei.gateway
├── config/                // 配置类
│   ├── properties/          // 各种配置属性类
│   └── GatewayConfig.java  // 网关主配置
├── filter/                // 请求过滤器
│   ├── AuthFilter.java     // 认证过滤器
│   ├── BlackListUrlFilter.java // 黑名单过滤器
│   ├── GlobalLogFilter.java // 全局日志过滤器
│   └── ...                // 其他过滤器
├── handler/               // 异常处理器
│   ├── GatewayExceptionHandler.java // 统一异常处理
│   └── SentinelFallbackHandler.java // Sentinel 回调处理
├── utils/                 // 工具类
│   └── WebFluxUtils.java  // WebFlux 工具方法
└── RuoYiGatewayApplication.java // 启动类
```


---

## 🧠 技术栈与架构设计

### 技术栈
| 技术 | 用途 |
|------|------|
| Spring Cloud Gateway | 网关核心框架 |
| Sa-Token | 权限控制 |
| Sentinel | 流量控制与限流 |
| Nacos | 配置中心与服务发现 |
| WebFlux | 响应式编程模型 |
| Hutool | 工具类库 |

### 架构图（文字描述）

```
[客户端] → [Nginx/LB] → [Spring Cloud Gateway]
         ↓                     ↑
       路由转发             白名单过滤
                             ↓
                         登录鉴权 (Sa-Token)
                             ↓
                       请求日志记录
                             ↓
                        自定义限流
                             ↓
                      业务微服务
```


---

## 🔧 配置详解

### 1. `application.yml`
```yaml
server:
  port: 8080
spring:
  application:
    name: ruoyi-gateway
  cloud:
    gateway:
      requestLog: true # 是否开启请求日志
```


### 2. `CustomGatewayProperties.java`
- 控制是否启用请求日志 (`requestLog`)。
- 用于动态刷新配置。

### 3. `IgnoreWhiteProperties.java`
- 定义放行的白名单路径列表。
- 示例配置：
```yaml
security:
  ignore:
    whites:
      - /login
      - /auth/code
```


---

## ✅ 总结

`pei-gateway` 是一个典型的 Spring Cloud Gateway 应用，具备完整的 API 网关能力：

- **统一鉴权**：使用 Sa-Token 实现安全访问控制。
- **黑白名单管理**：灵活配置黑名单和放行路径。
- **限流保护**：集成 Sentinel 提供熔断与限流。
- **日志追踪**：记录请求参数、路径、耗时等信息。
- **异常统一处理**：捕获并处理所有异常，返回标准响应格式。
