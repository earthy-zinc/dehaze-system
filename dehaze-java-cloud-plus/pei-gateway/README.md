# pei-gateway 网关模块说明文档

## 1. 总体说明

### 核心职责

`pei-gateway` 是基于 Spring Cloud Gateway 构建的微服务网关模块，作为整个微服务架构的统一入口点。其主要职责包括：

1. **请求路由**：根据配置的路由规则将请求转发到相应的后端微服务
2. **身份认证**：统一验证请求的合法性，解析并验证 Token
3. **跨域处理**：解决前端跨域访问问题
4. **灰度发布**：基于请求头实现服务实例的版本选择
5. **访问日志**：记录详细的请求和响应信息
6. **异常处理**：统一处理网关层异常并返回标准错误响应

### 边界声明

**本模块负责：**
- 所有进入微服务架构的 HTTP 请求统一入口处理
- 请求的路由、认证、日志记录等通用功能
- 跨域问题的统一解决
- 灰度发布策略的实现

**本模块不负责：**
- 具体业务逻辑的处理
- 数据持久化操作
- 业务异常的具体处理（由各微服务自行处理）

---

## 2. 需求与背景

### 业务动机

在微服务架构中，每个服务都有独立的部署和接口，如果直接暴露给前端或第三方调用，会面临以下问题：

1. **安全性不足**：无法对所有请求进行统一的身份验证和权限控制
2. **请求管理困难**：多个服务的 API 分散，缺乏统一的路由和转发机制
3. **跨域问题**：不同服务之间可能存在跨域限制，前端调用复杂
4. **日志和监控缺失**：无法集中收集所有服务的访问日志，影响运维效率
5. **负载均衡和灰度发布**：缺乏统一的流量控制策略

为了解决这些问题，引入了 `pei-gateway` 网关模块。

### 技术驱动因素

1. **统一入口**：通过网关提供统一的访问入口，简化客户端调用
2. **安全控制**：在网关层统一进行身份认证，确保只有合法用户才能访问受保护的资源
3. **负载均衡**：结合 Nacos 和 LoadBalancer 实现服务发现与负载均衡
4. **灰度发布**：支持根据请求头中的 version 字段实现灰度发布
5. **可观测性**：记录详细的访问日志，便于后续分析和问题排查

### 需求映射

| 用户需求 | 技术实现 | 代码依据 |
|---------|---------|---------|
| 统一访问入口 | Spring Cloud Gateway | GatewayServerApplication.java |
| 身份认证 | TokenAuthenticationFilter | TokenAuthenticationFilter.java |
| 跨域处理 | CorsFilter + CorsResponseHeaderFilter | CorsFilter.java |
| 灰度发布 | GrayReactiveLoadBalancerClientFilter + GrayLoadBalancer | GrayReactiveLoadBalancerClientFilter.java |
| 日志记录 | AccessLogFilter | AccessLogFilter.java |
| 异常处理 | GlobalExceptionHandler | GlobalExceptionHandler.java |

---

## 3. 功能与非功能需求分析

### 功能性需求

1. **路由转发**：根据配置的路由规则将请求转发到对应的微服务
2. **身份认证**：验证请求携带的 Token 合法性
3. **跨域支持**：处理跨域请求并添加相应响应头
4. **灰度发布**：基于请求头中的 version 字段选择服务实例
5. **日志记录**：记录请求和响应的详细信息
6. **异常处理**：统一处理网关层异常并返回标准错误响应

### 非功能性需求

1. **性能**：作为所有请求的入口，需要具备高并发处理能力
2. **安全**：在网关层统一进行身份认证，保护后端服务
3. **可观测性**：详细记录访问日志，便于监控和问题排查
4. **可靠性**：具备良好的容错能力，异常情况下能返回标准错误响应

---

## 4. 技术栈与依赖解析

### 核心技术栈

- **Spring Cloud Gateway**: 基于 Spring Boot 3.4 的网关框架
- **Spring Cloud LoadBalancer**: 客户端负载均衡
- **Nacos**: 服务注册与发现
- **Reactor**: 响应式编程模型

### 核心依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
```

该依赖引入了 Spring Cloud Gateway 的核心功能，包括路由、过滤、负载均衡等。

```xml
<dependency>
    <groupId>com.pei</groupId>
    <artifactId>pei-module-system-api</artifactId>
    <version>${revision}</version>
</dependency>
```

该依赖提供了系统模块的 API 接口定义，用于网关与系统服务的交互。

---

## 5. 架构设计

### 5.1 分层结构

```
com.pei.dehaze.gateway
├── filter/               // 过滤器类，用于实现各种请求拦截逻辑
│   ├── cors/              // 跨域相关过滤器
│   ├── grey/              // 灰度发布相关过滤器
│   ├── logging/           // 访问日志记录
│   └── security/          // 安全认证相关过滤器
├── handler/              // 异常处理器，统一处理全局异常
├── jackson/              // Jackson 序列化配置，处理 JSON 格式数据
├── route/                // 动态路由配置（可从 Nacos 获取）
├── util/                 // 工具类，如 IP 获取、租户识别等
└── GatewayServerApplication.java // 启动类
```

### 5.2 组件交互图

```mermaid
graph TB
    A[客户端] --> B[CorsFilter]
    B --> C[TokenAuthenticationFilter]
    C --> D[GrayReactiveLoadBalancerClientFilter]
    D --> E[目标微服务]
    E --> F[AccessLogFilter]
    F --> G[客户端]
    H[GlobalExceptionHandler] --> G
```

### 5.3 关键流程时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant CorsFilter
    participant TokenAuthenticationFilter
    participant GrayReactiveLoadBalancerClientFilter
    participant AccessLogFilter
    participant BackendService as 后端服务
    participant GlobalExceptionHandler
    
    Client->>CorsFilter: 发送 HTTP 请求
    CorsFilter->>CorsFilter: 处理跨域
    CorsFilter->>TokenAuthenticationFilter: 继续请求链
    TokenAuthenticationFilter->>TokenAuthenticationFilter: 验证 Token
    TokenAuthenticationFilter->>GrayReactiveLoadBalancerClientFilter: 继续请求链
    GrayReactiveLoadBalancerClientFilter->>BackendService: 转发请求
    BackendService->>AccessLogFilter: 返回响应
    AccessLogFilter->>AccessLogFilter: 记录日志
    AccessLogFilter->>Client: 返回响应
    
    GlobalExceptionHandler->>GlobalExceptionHandler: 处理异常
    GlobalExceptionHandler->>Client: 返回错误响应
```

---

## 6. 核心实现详解

### 6.1 路由配置

路由配置在 application.yaml 文件中定义，示例如下：

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: system-admin-api
          uri: grayLb://system-server
          predicates:
            - Path=/admin-api/system/**
          filters:
            - RewritePath=/admin-api/system/v3/api-docs, /v3/api-docs
```

该配置定义了将 `/admin-api/system/**` 路径的请求路由到 `system-server` 服务，并重写路径以适配 Swagger 文档访问。

### 6.2 身份认证实现

TokenAuthenticationFilter 负责身份认证：

1. 从请求头中提取 `Authorization` 字段
2. 调用 OAuth2 服务验证 Token 合法性
3. 将用户信息写入请求上下文

核心代码逻辑：
```java
String authorization = SecurityFrameworkUtils.obtainAuthorization(exchange);
if (StrUtil.isNotEmpty(authorization)) {
    // 调用远程服务验证 Token
    CommonResult<OAuth2AccessTokenCheckRespDTO> accessToken = oauth2TokenApi.checkToken(authorization);
    if (accessToken.isSuccess()) {
        // 将用户信息写入 exchange 属性和请求头
        exchange.getAttributes().put(WebFrameworkUtils.LOGIN_USER_KEY, loginUser);
        request.mutate().header(WebFrameworkUtils.LOGIN_USER_HEADER, JsonUtils.toJsonString(loginUser)).build();
    }
}
```

### 6.3 灰度发布实现

灰度发布通过 GrayReactiveLoadBalancerClientFilter 和 GrayLoadBalancer 实现：

1. 从请求头中提取 version 字段
2. 根据版本信息筛选匹配的服务实例
3. 若无匹配项，则使用默认策略选择服务实例

核心代码逻辑：
```java
private Response<ServiceInstance> getInstanceResponse(List<ServiceInstance> instances, HttpHeaders headers) {
    String version = headers.getFirst(VERSION);
    List<ServiceInstance> chooseInstances;
    if (StrUtil.isEmpty(version)) {
        chooseInstances = instances;
    } else {
        chooseInstances = CollectionUtils.filterList(instances,
                instance -> version.equals(instance.getMetadata().get("version")));
    }
    return new DefaultResponse(NacosBalancer.getHostByRandomWeight3(chooseInstances));
}
```

### 6.4 访问日志实现

AccessLogFilter 负责记录访问日志：

1. 拦截请求和响应内容
2. 构建 AccessLog 对象记录详细信息
3. 支持打印到日志或发送至远程服务存储

核心代码逻辑：
```java
private Mono<Void> writeWith(Publisher<? extends DataBuffer> body) {
    if (body instanceof Flux) {
        return super.writeWith(Flux.from(body).buffer().map(dataBuffers -> {
            byte[] content = readContent(dataBuffers);
            String responseResult = new String(content, StandardCharsets.UTF_8);
            gatewayLog.setResponseBody(responseResult);
            return bufferFactory.wrap(content);
        }));
    }
    return super.writeWith(body);
}
```

### 6.5 全局异常处理

GlobalExceptionHandler 统一处理所有异常：

1. 捕获网关层的所有异常
2. 返回统一格式的 JSON 错误响应

核心代码逻辑：
```java
@Order(-1)
@Slf4j
public class GlobalExceptionHandler implements ErrorWebExceptionHandler {
    @Override
    public Mono<Void> handle(ServerWebExchange exchange, Throwable ex) {
        // 处理异常并返回统一格式的错误响应
        CommonResult<?> result = CommonResult.error(INTERNAL_SERVER_ERROR.getCode(), INTERNAL_SERVER_ERROR.getMsg());
        return WebFrameworkUtils.writeJSON(exchange.getResponse(), result);
    }
}
```

---

## 7. 网关模块完整工作图解

### 时序图

```mermaid
sequenceDiagram
    participant Client
    participant CorsFilter
    participant TokenAuthenticationFilter
    participant GrayReactiveLoadBalancerClientFilter
    participant AccessLogFilter
    participant RouteHandler
    participant ServiceInstance
    participant BackendService
    participant GlobalExceptionHandler
    Client ->> CorsFilter: 发送 HTTP 请求（可能跨域）
    CorsFilter ->> CorsFilter: 设置 Access-Control-* 响应头
    CorsFilter ->> TokenAuthenticationFilter: 继续请求链
    TokenAuthenticationFilter ->> SecurityFrameworkUtils: 提取 Authorization Header 中的 Token
    SecurityFrameworkUtils -->> TokenAuthenticationFilter: 返回 Token 字符串
    TokenAuthenticationFilter ->> TokenAuthenticationFilter: 调用远程服务验证 Token 有效性
    TokenAuthenticationFilter ->> SecurityFrameworkUtils: 设置 LoginUser 到 Exchange 属性和 Request Header
    SecurityFrameworkUtils -->> TokenAuthenticationFilter: 完成用户信息设置
    TokenAuthenticationFilter ->> GrayReactiveLoadBalancerClientFilter: 继续请求链
    GrayReactiveLoadBalancerClientFilter ->> GrayLoadBalancer: 获取当前请求的 version 标签
    GrayLoadBalancer -->> GrayReactiveLoadBalancerClientFilter: 返回匹配的服务实例（按 metadata.version 过滤）
    GrayReactiveLoadBalancerClientFilter ->> BackendService: 将请求转发到匹配的后端服务
    BackendService -->> GrayReactiveLoadBalancerClientFilter: 返回响应数据
    GrayReactiveLoadBalancerClientFilter ->> AccessLogFilter: 继续处理响应链
    AccessLogFilter ->> WebFrameworkUtils: 记录访问日志（包含 userId, userType, tenantId, requestUrl, responseCode 等）
    WebFrameworkUtils -->> AccessLogFilter: 日志记录完成
    AccessLogFilter ->> Client: 返回 HTTP 响应

    opt 异常情况
        CorsFilter ->> GlobalExceptionHandler: 如果出现异常
        TokenAuthenticationFilter ->> GlobalExceptionHandler: Token 验证失败或过期
        GrayReactiveLoadBalancerClientFilter ->> GlobalExceptionHandler: 找不到匹配服务实例
        AccessLogFilter ->> GlobalExceptionHandler: 日志记录失败
        GlobalExceptionHandler -->> Client: 返回标准化错误码（如 401、500）
    end
```

---

#### 流程说明与逻辑解析

1. **客户端发送请求**

- 客户端向网关发起 HTTP 请求，例如：
  ```http
  GET /api/user HTTP/1.1
  Host: gateway.example.com
  Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  version: v2
  ```

2. **跨域处理 (`CorsFilter`)`

- 网关注册了两个 CORS 相关 Filter：
    - `CorsFilter`: 添加标准的跨域响应头，如 `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`
    - `CorsResponseHeaderFilter`: 解决 Spring Cloud Gateway 默认添加多个 `Access-Control-Allow-Origin` 导致浏览器拒绝的问题
- 如果是 `OPTIONS` 请求，则直接返回 `200 OK` 并结束流程

3. **身份认证 (`TokenAuthenticationFilter`)`

- 网关对请求进行统一的身份认证：
    - 从 `Authorization` Header 提取 Token
    - 使用 `WebClient` 调用 `/oauth2/check-token` 接口验证 Token 合法性
    - 将登录用户信息（userId、userType、tenantId）写入 `exchange.getAttributes()` 和 `request.header("login-user")`
- 如果 Token 无效或过期，仍会继续请求链，由下游服务做权限控制

4. **灰度路由 (`GrayReactiveLoadBalancerClientFilter`)`

- 网关支持基于请求头中的 `version` 实现灰度发布。
- 使用自定义负载均衡器 `GrayLoadBalancer`：
    - 从请求头提取 `version`
    - 匹配 Nacos 注册中心中对应版本的服务实例
    - 若无匹配项，则使用默认策略（随机加权选择）
- 最终将请求转发到匹配的服务实例

5. **日志记录 (`AccessLogFilter`)`

- 记录完整的请求和响应内容，包括：
    - 请求方法、URL、QueryParams、RequestBody
    - 用户 ID、用户类型、租户 ID
    - 响应体、响应状态码、执行耗时
- 支持两种方式输出：
    - 控制台打印（开发环境）
    - 异步写入数据库（生产环境）

6. **请求转发与业务处理**

- 请求通过上述过滤器链后，最终被转发到目标服务（如 system-server、member-server 等）。
- 目标服务处理完成后，返回响应给网关。

7. **响应返回客户端**

- 网关将后端服务的响应返回给客户端。
- 所有响应都经过 `AccessLogFilter` 记录完整的访问日志。

8. **全局异常处理 (`GlobalExceptionHandler`)`

- 网关统一捕获所有异常：
    - 如 Token 无效、找不到服务实例等
- 返回统一格式的 JSON 错误响应，如：
  ```json
  {
    "code": 401,
    "msg": "Unauthorized",
    "data": null
  }
  ```

#### 各组件作用详解

| 组件名                                    | 功能               | 关键作用                              |
|----------------------------------------|------------------|-----------------------------------|
| `CorsFilter`                           | 处理跨域问题           | 添加标准 CORS 响应头，解决前端跨域限制            |
| `TokenAuthenticationFilter`            | Token 验证与用户上下文注入 | 从请求头提取 Token，验证合法性，注入用户信息到 Header |
| `GrayReactiveLoadBalancerClientFilter` | 灰度发布支持           | 根据 `version` 请求头筛选服务实例，实现灰度路由     |
| `AccessLogFilter`                      | 请求日志记录           | 记录请求参数、响应内容、执行时间等信息               |
| `GlobalExceptionHandler`               | 全局异常处理           | 捕获并统一返回异常信息，避免暴露堆栈信息              |

---

#### 核心交互流程总结

| 步骤 | 操作       | 说明                     |
|----|----------|------------------------|
| 1  | 客户端发起请求  | 包含 Token、version、路径等信息 |
| 2  | CORS 处理  | 设置跨域响应头，允许前端调用         |
| 3  | Token 验证 | 提取 Token，验证有效性，设置用户信息  |
| 4  | 灰度路由     | 根据 version 请求头匹配服务实例   |
| 5  | 日志记录     | 记录完整请求和响应信息            |
| 6  | 请求转发     | 路由到对应微服务，执行业务逻辑        |
| 7  | 响应返回     | 返回后端结果给客户端             |
| 8  | 异常处理     | 捕获所有异常，返回标准化错误信息       |

---

### 流程图

```mermaid
graph TD
    A[客户端发起请求] --> B{是否为跨域请求?}
    B -- 是 --> C[添加 CORS 响应头]
    B -- 否 --> D[继续请求链]
    C --> E[TenantWebFilter 设置租户上下文]
    D --> E
    E --> F[TokenAuthenticationFilter 验证 Token]
    F --> G{Token 是否有效?}
    G -- 有效 --> H[设置 LoginUser 到 Header 和 Exchange]
    G -- 无效 --> I[跳过用户设置, 继续请求链]

H --> J[GrayReactiveLoadBalancerClientFilter 路由选择]
I --> J

J --> K{是否存在 version 请求头?}
K -- 存在 --> L[GrayLoadBalancer 根据 version 匹配服务实例]
K -- 不存在 --> M[使用默认负载均衡策略选择服务实例]

L --> N[转发请求到匹配的服务实例]
M --> N

N --> O[BackendService 执行业务逻辑]
O --> P[BackendService 返回响应]

P --> Q[AccessLogFilter 记录访问日志]
Q --> R[GlobalExceptionHandler 处理异常]

R --> S{是否有异常?}
S -- 是 --> T[返回标准化错误信息 CommonResult.error]
S -- 否 --> U[返回正常响应数据]

T --> V[客户端收到错误响应]
U --> V[客户端收到正常响应]

V --> W[流程结束]
```

---

#### 图解说明与逻辑细化

1. **客户端发起请求**

- 客户端向网关发起 HTTP 请求。
- 示例：
  ```http
  GET /api/user HTTP/1.1
  Host: gateway.example.com
  Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  version: v2
  ```

2. **跨域判断与处理**

- 如果是跨域请求（CORS），触发 `CorsFilter`
- 添加标准的跨域响应头，如 `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`
- 同时通过 `CorsResponseHeaderFilter` 解决重复 `Access-Control-Allow-Origin` 导致浏览器拒绝的问题

3. **租户上下文识别**

- `TenantWebFilter` 解析请求头中的 `X-Tenant-ID` 或从用户信息中获取租户 ID
- 设置到 `TenantContextHolder`

4. **Token 验证**

- `TokenAuthenticationFilter` 提取 `Authorization` 请求头中的 Token
- 调用 `/oauth2/check-token` 接口验证 Token 合法性
- 若 Token 有效，将用户信息（userId、userType、tenantId）注入到：
    - `exchange.getAttributes()` 中
    - 请求 Header 的 `login-user` 字段中

5. **灰度路由选择**

- `GrayReactiveLoadBalancerClientFilter` 触发
- 从请求头提取 `version` 字段
- 使用 `GrayLoadBalancer` 进行筛选：
    - 如果存在 `version`，则匹配 metadata.version 相同的服务实例
    - 否则，使用随机 + 权重策略选择服务实例

6. **请求转发至后端服务**

- 将请求转发到目标微服务（如 system-server、member-server）
- 支持服务发现和负载均衡（Nacos + LoadBalancer）

7. **后端服务执行业务逻辑**

- 微服务接收到请求后，执行具体业务逻辑
- 可能涉及数据库查询、缓存读写、消息队列调用等操作

8. **响应返回网关**

- 后端服务将响应返回给网关
- 响应体可能包含 JSON 数据、状态码、自定义 Header 等

9. **记录访问日志**

- `AccessLogFilter` 拦截响应内容
- 构建完整的 `AccessLog` 对象
- 日志可打印到控制台或异步写入数据库

10. **全局异常处理**

- `GlobalExceptionHandler` 统一捕获所有异常
- 包括：
    - Token 验证失败
    - 服务不可用
    - 内部服务器错误
- 返回统一格式的 JSON 错误响应，例如：
  ```json
  {
    "code": 401,
    "msg": "Unauthorized",
    "data": null
  }
  ```

11. **响应返回客户端**

- 最终响应返回给客户端
- 包含状态码、JSON 数据、必要的响应头（如 Content-Type）

12. **流程结束**

- 整个请求处理完成
- 所有线程上下文变量被清理（如 TenantContextHolder）

---

#### 各阶段作用总结

| 步骤    | 关键组件                                 | 功能描述                              |
|-------|--------------------------------------|-----------------------------------|
| 1     | Client                               | 发起 HTTP 请求，携带 Token、version、路径等信息 |
| 2~3   | CorsFilter                           | 处理跨域问题，允许前端访问                     |
| 4     | TenantWebFilter                      | 设置当前线程的租户上下文，用于多租户隔离              |
| 5~7   | TokenAuthenticationFilter            | 验证 Token 合法性，设置登录用户信息             |
| 8~10  | GrayReactiveLoadBalancerClientFilter | 根据 version 请求头动态选择服务实例，实现灰度发布     |
| 11~13 | AccessLogFilter                      | 记录完整的请求日志，便于监控与审计                 |
| 14~16 | GlobalExceptionHandler               | 统一处理异常，避免暴露堆栈信息                   |
| 17~18 | GatewayServerApplication             | Spring Boot 主程序，启动网关服务            |

---

## 8. 部署与运维

### 启动方式

通过 GatewayServerApplication 启动类启动服务：

```java
@SpringBootApplication
public class GatewayServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(GatewayServerApplication.class, args);
    }
}
```

### Docker 部署

使用提供的 Dockerfile 进行容器化部署：

```dockerfile
FROM eclipse-temurin:21-jre

RUN mkdir -p /pei-gateway
WORKDIR /pei-gateway
COPY ./target/pei-gateway.jar app.jar

ENV TZ=Asia/Shanghai JAVA_OPTS="-Xms256m -Xmx256m"

EXPOSE 48080

CMD ["sh", "-c", "exec java ${JAVA_OPTS} -Djava.security.egd=file:/dev/./urandom -jar app.jar"]
```

### 日志配置

日志配置在 logback-spring.xml 中定义，支持控制台和文件两种输出方式：

```xml
<configuration>
    <!-- 控制台 Appender -->
    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder class="ch.qos.logback.core.encoder.LayoutWrappingEncoder">
            <layout class="org.apache.skywalking.apm.toolkit.log.logback.v1.x.TraceIdPatternLogbackLayout">
                <pattern>${PATTERN_DEFAULT}</pattern>
            </layout>
        </encoder>
    </appender>
    
    <!-- 文件 Appender -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <!-- 文件滚动策略和编码配置 -->
    </appender>
</configuration>
```

---

## 9. 总结

`pei-gateway` 网关模块作为微服务架构的统一入口，承担着路由转发、身份认证、跨域处理、灰度发布、日志记录等重要职责。通过合理的设计和实现，为整个系统提供了安全、可靠、可观察的流量入口，有效简化了客户端与后端服务的交互复杂度。

| 模块                | 主要职责     | 关键实现                        |
|-------------------|----------|-----------------------------|
| `filter/cors`     | 解决跨域问题   | 设置 `Access-Control-*` 响应头   |
| `filter/grey`     | 实现灰度发布   | 自定义 `GrayLoadBalancer`      |
| `filter/logging`  | 记录访问日志   | `AccessLogFilter` 拦截请求/响应   |
| `filter/security` | Token 认证 | `TokenAuthenticationFilter` |
| `handler`         | 全局异常处理   | `GlobalExceptionHandler`    |
| `jackson`         | JSON 序列化 | 自定义 `NumberSerializer`      |
| `route`           | 动态路由     | 从 Nacos 加载 `routes` 配置      |
| `util`            | 工具类      | 提供 Token 解析、IP 获取等功能        |