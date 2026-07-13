### 现有 Java 基础设施/架构主要问题（基于当前文档与实现状态）

> 更新说明：原"消息队列能力缺失"项已删除，Java 端现已集成 RabbitMQ + Kafka（见 `dehaze-java/src/main/java/com/pei/dehaze/mq/`，含 RabbitMQPublisher/Consumer、KafkaLogProducer/Consumer、ExportTaskConsumer、DownloadTaskConsumer、AuditLogConsumer、ThumbnailTaskConsumer、SystemLogConsumer 等）。

- **缓存体系缺少 L1 与穿透/击穿防护**：当前仅有 Spring Cache + Redis（`7.5` 也表明 L1 与布隆过滤器是规划项），这意味着**热点数据访问延迟较高**且**高并发下的穿透/击穿风险**仍存在。

- **链路可观测性不足**：文档只提到 Actuator 与 Prometheus 指标（`17.1`），但缺少**TraceId 贯穿日志/链路追踪（OpenTelemetry/Zipkin/Jaeger）**的基础设施设计，当前日志只能做事后排查，**无法形成跨服务的全链路定位**；接口调用应统一透传 `X-Trace-ID`（兼容 `traceparent`/`sw8`）并在文档中明确请求头规范。

- **日志集中化未落地**：`10.4` 中的“Logback → Kafka → ES/S3”是规划项，当前仅本地文件输出，**缺乏集中检索、告警与审计管道**。

- **异步任务可恢复性不足**：现有 `@Async` 线程池未体现**任务状态持久化、失败重试、幂等保证**与**任务队列补偿机制**，任务稳定性依赖应用进程存活。
