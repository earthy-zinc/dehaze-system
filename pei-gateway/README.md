`pei-gateway` 是一个基于 Spring Cloud Gateway
的网关模块，作为微服务架构中的入口点。它的主要作用是集中处理请求的路由、安全认证、跨域配置、日志记录等功能。通过网关，可以统一管理各个微服务的访问入口，并提供通用的功能支持。

---

## 一、简介

在微服务架构中，每个服务都有自己的独立部署和接口，如果直接暴露给前端或第三方调用，会导致以下问题：

- **安全性不足**：无法对所有请求进行统一的身份验证和权限控制。
- **请求管理困难**：多个服务的 API 分散，缺乏统一的路由和转发机制。
- **跨域问题**：不同服务之间可能存在跨域限制，前端调用复杂。
- **日志和监控缺失**：无法集中收集所有服务的访问日志，影响运维效率。
- **负载均衡和灰度发布**：缺乏统一的流量控制策略，难以实现灵活的服务治理。

为了解决这些问题，引入了 `pei-gateway` 网关模块，它具备如下核心优势：

### ✅ 核心作用

1. **统一入口**：将所有服务的请求都经过网关，对外提供统一的访问入口。
2. **身份认证**：集成 Token 验证机制，确保只有合法用户才能访问受保护的资源。
3. **跨域处理**：统一解决跨域问题，简化前端开发。
4. **灰度发布**：支持根据请求头（如 version）动态选择服务实例，实现灰度测试。
5. **日志记录**：记录详细的访问日志，便于后续分析和排查问题。
6. **负载均衡**：结合 Nacos 和 LoadBalancer 实现服务发现与负载均衡。
7. **异常处理**：统一捕获并返回异常信息，提升系统健壮性。

---

## 二、模块结构说明

```
src/main/java/
└── com/pei/dehaze/gateway/
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

---

## 三、包结构详解

### 1️⃣ `filter` 包

#### 📁 `cors` 子包

- **功能**：处理跨域请求。
- **关键类**：
    - `CorsFilter`：设置响应头允许跨域访问。
    - `CorsResponseHeaderFilter`：修复 Spring Cloud Gateway 中重复 `Access-Control-Allow-Origin` 头的问题。

#### 📁 `grey` 子包

- **功能**：实现灰度发布，根据请求头（如 `version`）选择对应版本的服务实例。
- **关键类**：
    - `GrayLoadBalancer`：自定义负载均衡器，支持按版本匹配。
    - `GrayReactiveLoadBalancerClientFilter`：注册为全局过滤器，接管默认的负载均衡逻辑。

#### 📁 `logging` 子包

- **功能**：记录访问日志，包括请求参数、响应体、执行时间等。
- **关键类**：
    - `AccessLog`：日志实体类。
    - `AccessLogFilter`：全局过滤器，记录完整的 HTTP 请求日志。

#### 📁 `security` 子包

- **功能**：实现基于 Token 的身份认证。
- **关键类**：
    - `LoginUser`：封装登录用户信息。
    - `TokenAuthenticationFilter`：全局过滤器，解析 Token 并设置用户上下文。

---

### 2️⃣ `handler` 包

- **功能**：统一处理异常，返回标准化错误响应。
- **关键类**：
    - `GlobalExceptionHandler`：继承 `ErrorWebExceptionHandler`，捕获并处理所有异常，返回 `CommonResult` 格式的错误信息。

---

### 3️⃣ `jackson` 包

- **功能**：定制 Jackson 序列化规则，解决 Long 类型精度丢失、日期格式等问题。
- **关键类**：
    - `JacksonAutoConfiguration`：配置 Jackson 的序列化规则，如 Long 转 String、LocalDateTime 使用时间戳等。

---

### 4️⃣ `route` 包

- **功能**：动态路由配置，支持从 Nacos 获取最新的路由规则。
- **关键文件**：
    - `package-info.java`：占位符。
    - `dynamic/package-info.java`：描述如何从 Nacos 动态刷新路由。

---

### 5️⃣ `util` 包

- **功能**：工具类，提供通用辅助方法。
- **关键类**：
    - `SecurityFrameworkUtils`：处理 Token 解析、用户信息设置等。
    - `WebFrameworkUtils`：获取客户端 IP、写入 JSON 响应、构建路由信息等。
    - `EnvUtils`：处理环境变量、标签识别等。

---

## 四、核心代码实现原理

### 1️⃣ 路由配置 (`application.yaml`)

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

- **作用**：定义请求路径 `/admin-api/system/**` 被路由到 `system-server` 服务。
- **RewritePath**：重写路径，适配 Swagger 文档访问。

---

### 2️⃣ 身份认证 (`TokenAuthenticationFilter`)

```java
public class TokenAuthenticationFilter implements GlobalFilter, Ordered {
    private final WebClient webClient;
    private final LoadingCache<KeyValue<Long, String>, LoginUser> loginUserCache;

    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = SecurityFrameworkUtils.obtainAuthorization(exchange);
        return getLoginUser(exchange, token).defaultIfEmpty(LOGIN_USER_EMPTY)
                .flatMap(user -> {
                    if (user == LOGIN_USER_EMPTY) {
                        return chain.filter(exchange);
                    }
                    SecurityFrameworkUtils.setLoginUser(exchange, user);
                    ServerWebExchange newExchange = exchange.mutate()
                            .request(builder -> SecurityFrameworkUtils.setLoginUserHeader(builder, user))
                            .build();
                    return chain.filter(newExchange);
                });
    }
}
```

- **流程**：
    1. 从 Header 中提取 Token。
    2. 查询缓存或远程校验 Token 是否有效。
    3. 如果有效，设置用户信息到 Header 并转发请求；否则继续链式调用。

---

### 3️⃣ 灰度发布 (`GrayLoadBalancer`)

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

- **流程**：
    1. 从请求头获取 `version`。
    2. 匹配 metadata 中包含相同 `version` 的服务实例。
    3. 若无匹配，则使用默认权重随机选择实例。

---

### 4️⃣ 日志记录 (`AccessLogFilter`)

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

- **流程**：
    1. 拦截请求和响应内容。
    2. 构建 `AccessLog` 对象，记录详细请求信息。
    3. 支持打印到日志或发送至远程服务存储。

---

## 五、网关模块完整工作图解

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

#### 📚 流程说明与逻辑解析

1. **客户端发送请求**

- 客户端向网关发起 HTTP 请求，例如：
  ```http
  GET /api/user HTTP/1.1
  Host: gateway.example.com
  Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  version: v2
  ```

2. **跨域处理 (`CorsFilter`)**

- 网关注册了两个 CORS 相关 Filter：
    - `CorsFilter`: 添加标准的跨域响应头，如 `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`
    - `CorsResponseHeaderFilter`: 解决 Spring Cloud Gateway 默认添加多个 `Access-Control-Allow-Origin` 导致浏览器拒绝的问题
- 如果是 `OPTIONS` 请求，则直接返回 `200 OK` 并结束流程

3. **身份认证 (`TokenAuthenticationFilter`)**

- 网关对请求进行统一的身份认证：
    - 从 `Authorization` Header 提取 Token
    - 使用 `WebClient` 调用 `/oauth2/check-token` 接口验证 Token 合法性
    - 将登录用户信息（userId、userType、tenantId）写入 `exchange.getAttributes()` 和 `request.header("login-user")`
- 如果 Token 无效或过期，仍会继续请求链，由下游服务做权限控制

4. **灰度路由 (`GrayReactiveLoadBalancerClientFilter`)**

- 网关支持基于请求头中的 `version` 实现灰度发布。
- 使用自定义负载均衡器 `GrayLoadBalancer`：
    - 从请求头提取 `version`
    - 匹配 Nacos 注册中心中对应版本的服务实例
    - 若无匹配项，则使用默认策略（随机加权选择）
- 最终将请求转发到匹配的服务实例

5. **日志记录 (`AccessLogFilter`)**

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

8. **全局异常处理 (`GlobalExceptionHandler`)**

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

#### 🔍 各组件作用详解

| 组件名                                    | 功能               | 关键作用                              |
|----------------------------------------|------------------|-----------------------------------|
| `CorsFilter`                           | 处理跨域问题           | 添加标准 CORS 响应头，解决前端跨域限制            |
| `TokenAuthenticationFilter`            | Token 验证与用户上下文注入 | 从请求头提取 Token，验证合法性，注入用户信息到 Header |
| `GrayReactiveLoadBalancerClientFilter` | 灰度发布支持           | 根据 `version` 请求头筛选服务实例，实现灰度路由     |
| `AccessLogFilter`                      | 请求日志记录           | 记录请求参数、响应内容、执行时间等信息               |
| `GlobalExceptionHandler`               | 全局异常处理           | 捕获并统一返回异常信息，避免暴露堆栈信息              |

---

#### 🔄 核心交互流程总结

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

#### 📌 图解说明与逻辑细化

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

#### 🔍 各阶段作用总结

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

## 六、总结

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
