`pei-spring-boot-starter-mq` 是一个 **统一的消息队列封装模块（Message Queue Extension Module）**，其核心作用是为企业级应用提供统一的 MQ 操作接口，并支持以下消息中间件：

- Redis Stream/PubSub
- RabbitMQ
- RocketMQ
- Kafka

该模块基于 Spring Boot + Spring Data Redis + Spring AMQP 实现，提供了拦截器、重发机制、日志记录等高级功能，适用于电商订单系统、会员中心、AI 大模型任务处理、CRM 客户管理、ERP 库存同步等需要高可用消息队列的场景。

---

## ✅ 模块概述

### 🎯 模块定位
- **目标**：构建统一的消息队列封装层，支持：
    - 消息发送与消费的统一接口
    - 支持多种 MQ 类型（Redis、RabbitMQ、RocketMQ、Kafka）
    - 提供拦截器机制，实现多租户、日志记录、事务控制等功能
    - 自动清理过期消息、自动重试未完成消息
- **应用场景**：
    - 订单状态变更通知
    - 用户行为埋点上报
    - AI 模型训练异步触发
    - CRM 客户数据同步
    - ERP 库存更新广播

### 🧩 技术栈依赖
- **基础框架**：Spring Boot 3.4 + Java 17
- **消息队列**：
    - Redis Stream / PubSub（默认集成）
    - RabbitMQ（可选）
    - RocketMQ（可选）
    - Kafka（可选）
- **序列化工具**：Jackson JSON
- **定时任务**：@Scheduled
- **分布式锁**：Redisson

---

## 📁 目录结构说明

```
src/main/java/
└── com/pei/dehaze/framework/mq/
    ├── rabbitmq/                  // RabbitMQ 集成
    │   └── config/                // RabbitMQ 配置类
    └── redis/                     // Redis 集成
        ├── config/                // Redis MQ 的配置类（生产者、消费者）
        ├── core/                  // 核心逻辑实现
        │   ├── interceptor/       // 拦截器
        │   ├── job/               // 定时任务（消息重发、清理）
        │   ├── message/           // 消息基类定义
        │   ├── pubsub/            // Redis Pub/Sub 广播消费
        │   └── stream/            // Redis Stream 集群消费
        └── RedisMQTemplate.java   // 消息操作模板类
```


---

## 🔍 关键包详解

### 1️⃣ `rabbitmq.config` 包 —— RabbitMQ 配置类

#### 示例：`PeiRabbitMQAutoConfiguration.java`
```java
@Bean
public MessageConverter createMessageConverter() {
    return new Jackson2JsonMessageConverter();
}
```


- **作用**：为 RabbitMQ 设置消息序列化方式（JSON）。
- **关键逻辑**：
    - 使用 Jackson 进行消息体的序列化和反序列化
    - 支持自定义消息转换器扩展
- **设计模式**：
    - 策略模式（支持不同消息格式）

---

### 2️⃣ `redis.config` 包 —— Redis MQ 配置类

#### 示例：`PeiRedisMQProducerAutoConfiguration.java`
```java
@Bean
public RedisMQTemplate redisMQTemplate(StringRedisTemplate redisTemplate, List<RedisMessageInterceptor> interceptors) {
    RedisMQTemplate redisMQTemplate = new RedisMQTemplate(redisTemplate);
    interceptors.forEach(redisMQTemplate::addInterceptor);
    return redisMQTemplate;
}
```


- **作用**：初始化 Redis 消息发送模板。
- **关键逻辑**：
    - 注册 `RedisMQTemplate` Bean
    - 添加所有拦截器（用于日志记录、多租户隔离等）
- **优势**：
    - 支持插件式拓展
    - 统一消息发送入口

#### 示例：`PeiRedisMQConsumerAutoConfiguration.java`
```java
@Bean
public RedisMessageListenerContainer redisMessageListenerContainer(...) {
    container.addMessageListener(listener, new ChannelTopic(listener.getChannel()));
    return container;
}
```


- **作用**：注册 Redis Pub/Sub 和 Stream 的监听容器。
- **关键逻辑**：
    - 自动注册 `AbstractRedisChannelMessageListener`
    - 自动创建 Consumer Group
    - 支持手动 ACK，防止消息丢失
- **设计模式**：
    - 工厂模式（StreamMessageListenerContainer）
    - 观察者模式（监听器）

---

### 3️⃣ `redis.core.interceptor` 包 —— 拦截器接口

#### 示例：`RedisMessageInterceptor.java`
```java
public interface RedisMessageInterceptor {
    default void sendMessageBefore(AbstractRedisMessage message) {}
    default void sendMessageAfter(AbstractRedisMessage message) {}
    default void consumeMessageBefore(AbstractRedisMessage message) {}
    default void consumeMessageAfter(AbstractRedisMessage message) {}
}
```


- **作用**：在消息发送/消费前后插入钩子逻辑。
- **典型用途**：
    - 多租户标识注入
    - 消息 ID 唯一性校验
    - 日志记录
    - 事务绑定
- **使用方式**：
  ```java
  @Component
  public class TenantRedisMessageInterceptor implements RedisMessageInterceptor {
      @Override
      public void sendMessageBefore(AbstractRedisMessage message) {
          message.addHeader("tenantId", TenantContextHolder.getTenantId());
      }
  }
  ```


---

### 4️⃣ `redis.core.job` 包 —— 消息队列定时任务

#### 示例：`RedisPendingMessageResendJob.java`
```java
@Scheduled(cron = "35 * * * * ?")
public void messageResend() {
    ops.pending(streamKey, groupName);
    if (lastDelivery > EXPIRE_TIME) {
        redisTemplate.opsForStream().add(...)// 重新投递
    }
}
```


- **作用**：定期扫描未被正常消费的消息并重新投递。
- **关键逻辑**：
    - 使用 `XPENDING` 获取未确认消息
    - 判断超时时间（默认 5 分钟）
    - 重新发送消息并 ACK
- **优势**：
    - 保证消息最终一致性
    - 防止因服务宕机导致消息丢失

#### 示例：`RedisStreamMessageCleanupJob.java`
```java
@Scheduled(cron = "0 0 * * * ?")
public void cleanup() {
    ops.trim(streamKey, MAX_COUNT, true);
}
```


- **作用**：定期清理 Redis Stream 中的历史消息。
- **关键逻辑**：
    - 使用 `XTRIM` 清理超过最大保留条数的消息
    - 默认保留最近 10000 条消息
- **优势**：
    - 控制内存占用
      :nodoc:

    - 防止 Redis 内存溢出

---

### 5️⃣ `redis.core.message` 包 —— 消息抽象类

#### 示例：`AbstractRedisMessage.java`
```java
public abstract class AbstractRedisMessage {
    private final Map<String, String> headers = new HashMap<>();
}
```


- **作用**：定义消息的通用结构。
- **关键逻辑**：
    - 支持 Header 扩展
    - 支持泛型消息类型
- **设计模式**：
    - 模板方法模式（定义消息结构）
    - 工厂模式（子类继承后自动识别）

---

### 6️⃣ `redis.core.pubsub` 包 —— Redis Pub/Sub 消费

#### 示例：`AbstractRedisChannelMessage.java`
```java
public abstract class AbstractRedisChannelMessage extends AbstractRedisMessage {
    public String getChannel() {
        return getClass().getSimpleName();
    }
}
```


- **作用**：定义 Pub/Sub 消息的通道名。
- **使用方式**：
  ```java
  public class OrderStatusChangeMessage extends AbstractRedisChannelMessage {
      public String getChannel() {
          return "order.status.changed";
      }
  }
  ```


#### 示例：`AbstractRedisChannelMessageListener.java`
```java
@Override
public void onMessage(Message message, byte[] bytes) {
    T messageObj = JsonUtils.parseObject(message.getBody(), messageType);
    this.onMessage(messageObj);
}
```


- **作用**：实现 Pub/Sub 消息的统一处理。
- **关键逻辑**：
    - 消息反序列化
    - 调用 `onMessage(T message)` 抽象方法
- **设计模式**：
    - 模板方法模式（固定流程 + 子类实现业务逻辑）
    - 观察者模式（监听 Channel）

---

### 7️⃣ `redis.core.stream` 包 —— Redis Stream 消费

#### 示例：`AbstractRedisStreamMessage.java`
```java
public abstract class AbstractRedisStreamMessage extends AbstractRedisMessage {
    public String getStreamKey() {
        return getClass().getSimpleName();
    }
}
```


- **作用**：定义 Stream 消息的 Key。
- **使用方式**：
  ```java
  public class InventoryUpdateMessage extends AbstractRedisStreamMessage {
      public String getStreamKey() {
          return "inventory.update";
      }
  }
  ```


#### 示例：`AbstractRedisStreamMessageListener.java`
```java
@Override
public void onMessage(ObjectRecord<String, String> message) {
    T messageObj = JsonUtils.parseObject(message.getValue(), messageType);
    this.onMessage(messageObj);
}
```


- **作用**：实现 Stream 消息的统一处理。
- **关键逻辑**：
    - 使用 `XREADGROUP` 读取消息
    - 支持手动 ACK
    - 支持消费者分组、负载均衡
- **设计模式**：
    - 模板方法模式（固定流程 + 子类实现业务逻辑）
    - 观察者模式（监听 Stream Key）

---

### 8️⃣ `RedisMQTemplate.java` —— 消息操作模板

#### 示例：`RedisMQTemplate.java`
```java
public <T extends AbstractRedisChannelMessage> void send(T message) {
    redisTemplate.convertAndSend(message.getChannel(), JsonUtils.toJsonString(message));
}
```


- **作用**：提供统一的消息发送 API。
- **关键逻辑**：
    - 支持 Pub/Sub 发送
    - 支持 Stream 发送
    - 支持拦截器调用（before → after）
- **优势**：
    - 统一 API 接口
    - 支持扩展新类型的 MQ

---

## 🧠 模块工作流程图解

### 1️⃣ 消息发送流程（Pub/Sub）
```mermaid
graph TD
    A[Service 层调用 RedisMQTemplate.send(...)] --> B[调用 sendMessageBefore 拦截器]
    B --> C[通过 convertAndSend 发送到 Redis Channel]
    C --> D[调用 sendMessageAfter 拦截器]
```


### 2️⃣ 消息消费流程（Pub/Sub）
```mermaid
graph TD
    A[RedisMessageListenerContainer 监听 Channel] --> B[调用 onMessage 方法]
    B --> C[调用 consumeMessageBefore 拦截器]
    C --> D[执行具体业务逻辑 onMessage(...)]
    D --> E[调用 consumeMessageAfter 拦截器]
```


### 3️⃣ 消息发送流程（Stream）
```mermaid
graph TD
    A[Service 层调用 RedisMQTemplate.send(...)] --> B[调用 sendMessageBefore 拦截器]
    B --> C[使用 opsForStream().add(...) 发送消息]
    C --> D[调用 sendMessageAfter 拦截器]
```


### 4️⃣ 消息消费流程（Stream）
```mermaid
graph TD
    A[StreamMessageListenerContainer 监听 Stream Key] --> B[调用 onMessage 方法]
    B --> C[调用 consumeMessageBefore 拦截器]
    C --> D[执行具体业务逻辑 onMessage(...)]
    D --> E[手动调用 acknowledge(...)]
    E --> F[调用 consumeMessageAfter 拦截器]
```


---

## 🧱 模块间关系图

```mermaid
graph TD
    A[Controller] --> B[Service 层调用 RedisMQTemplate.send(...)]
    B --> C[RedisMQTemplate]
    C --> D{消息类型}
    D -- Pub/Sub --> E[AbstractRedisChannelMessage]
    D -- Stream --> F[AbstractRedisStreamMessage]
    E --> G[AbstractRedisChannelMessageListener]
    F --> H[AbstractRedisStreamMessageListener]
    G --> I[onMessage(...)]
    H --> J[onMessage(...)]
    C --> K[拦截器链]
    K --> L[RedisMessageInterceptor]
```


---

## 🧩 模块功能总结

| 包名 | 功能 | 关键类 |
|------|------|--------|
| `rabbitmq.config` | RabbitMQ 配置 | `PeiRabbitMQAutoConfiguration` |
| `redis.config` | Redis MQ 配置 | `PeiRedisMQProducerAutoConfiguration` |
| `redis.core.interceptor` | 拦截器接口 | `RedisMessageInterceptor` |
| `redis.core.job` | 消息重发与清理 | `RedisPendingMessageResendJob` |
| `redis.core.message` | 消息基类 | `AbstractRedisMessage` |
| `redis.core.pubsub` | Pub/Sub 消息处理 | `AbstractRedisChannelMessage` |
| `redis.core.stream` | Stream 消息处理 | `AbstractRedisStreamMessage` |
| `redis.core.RedisMQTemplate` | 消息操作模板 | `RedisMQTemplate` |

---

## ✅ 模块实现原理详解

### 1️⃣ 消息发送实现流程
- **步骤**：
    1. Service 调用 `RedisMQTemplate.send(...)` 或 `send(...)`
    2. 调用 `sendMessageBefore(...)` 拦截器
    3. 使用 `convertAndSend(...)` 发送到指定 Channel 或 Stream Key
    4. 调用 `sendMessageAfter(...)` 拦截器
- **示例**：
  ```java
  @Service
  public class OrderService {
      @Resource
      private RedisMQTemplate redisMQTemplate;

      public void updateOrderStatus(Long orderId) {
          OrderStatusChangedMessage message = new OrderStatusChangedMessage();
          message.setOrderId(orderId);
          redisMQTemplate.send(message); // 发送 Pub/Sub 消息
      }
  }
  ```


### 2️⃣ 消息消费实现流程（Pub/Sub）
- **步骤**：
    1. `AbstractRedisChannelMessageListener` 注册到 RedisMessageListenerContainer
    2. 监听指定 Channel
    3. 收到消息后调用 `consumeMessageBefore(...)` 拦截器
    4. 调用 `onMessage(...)` 抽象方法处理业务
    5. 调用 `consumeMessageAfter(...)` 拦截器
- **示例**：
  ```java
  @Component
  public class OrderStatusChangedMessageListener extends AbstractRedisChannelMessageListener<OrderStatusChangedMessage> {
      @Override
      public void onMessage(OrderStatusChangedMessage message) {
          log.info("收到订单状态变更: {}", message.getOrderId());
          // 更新库存、推送通知等
      }
  }
  ```


### 3️⃣ 消息消费实现流程（Stream）
- **步骤**：
    1. `AbstractRedisStreamMessageListener` 注册到 StreamMessageListenerContainer
    2. 创建 Consumer Group
    3. 收到消息后调用 `consumeMessageBefore(...)` 拦截器
    4. 调用 `onMessage(...)` 抽象方法处理业务
    5. 手动调用 `acknowledge(...)` 确认消费完成
    6. 调用 `consumeMessageAfter(...)` 拦截器
- **示例**：
  ```java
  @Component
  public class InventoryUpdateMessageListener extends AbstractRedisStreamMessageListener<InventoryUpdateMessage> {
      @Override
      public void onMessage(InventoryUpdateMessage message) {
          inventoryService.decreaseStock(message.getSkuId(), message.getCount());
      }
  }
  ```


---

## 🧪 单元测试与异常处理

### 示例：`RedisPendingMessageResendJobTest.java`
```java
@Test
public void testMessageResend() {
    when(ops.pending(eq(streamKey), any())).thenReturn(pendingSummary);
    job.messageResend();
    verify(redisTemplate).add(...)
}
```


- **作用**：验证消息重发逻辑的正确性。
- **覆盖范围**：
    - 正常情况（消息超时）
    - 异常情况（无 pending 消息）
- **测试覆盖率建议**：80%+

---

## ✅ 建议改进方向

| 改进点 | 描述 |
|--------|------|
| ✅ 消息幂等性增强 | 当前仅支持拦截器，未来可结合数据库或 Redis 缓存实现幂等性校验 |
| ✅ 消息失败重试策略 | 当前只支持重发一次，未来可加入重试次数、延迟重试机制 |
| ✅ 消息日志追踪 | 可结合 MDC 实现消息级别的日志追踪 |
| ✅ 消息事务绑定 | 可将消息发送与数据库事务绑定，实现本地事务回滚 |
| ✅ 多语言支持 | 当前仅支持中文，未来可扩展英文、日文等 |

---

## 📌 总结

`pei-spring-boot-starter-mq` 模块实现了以下核心功能：

| 功能 | 技术实现 | 用途 |
|------|-----------|------|
| 消息发送 | RedisMQTemplate | 统一发送 Pub/Sub 和 Stream 消息 |
| 消息消费 | AbstractRedisChannelMessageListener + AbstractRedisStreamMessageListener | 支持广播和集群消费两种模式 |
| 拦截器 | RedisMessageInterceptor | 实现消息头注入、日志记录、多租户支持 |
| 消息重发 | RedisPendingMessageResendJob | 防止因服务崩溃导致消息丢失 |
| 消息清理 | RedisStreamMessageCleanupJob | 防止 Redis Stream 消息堆积 |
| 消息模板 | RedisMQTemplate | 提供统一的发送入口 |

它是一个轻量但功能完整的 MQ 模块，适用于电商订单通知、库存更新、用户行为埋点、AI 模型任务分发等场景。

如果你有具体某个类（如 `RedisMQTemplate`、`RedisPendingMessageResendJob`）想要深入了解，欢迎继续提问！
