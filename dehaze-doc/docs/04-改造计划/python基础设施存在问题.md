### 现有 Python 基础设施/架构主要问题（基于当前代码与文档状态）

- **操作日志中间件使用 `BaseHTTPMiddleware` 存在性能隐患**：`OperationLogMiddleware` 基于 Starlette `BaseHTTPMiddleware` 实现，该中间件会**将整个请求体读入内存**且无法与 StreamingResponse 良好配合（已在代码中做了特殊处理但仍有 edge case）。对于大文件上传场景，存在内存压力风险。应评估迁移到纯 ASGI 中间件实现。

- **链路可观测性不足**：虽然 Prometheus 四大类指标已实现（HTTP/GPU/推理/任务），但缺少 **TraceID 贯穿日志/链路追踪（OpenTelemetry）** 的基础设施。当前日志只能做事后排查，无法形成跨 Python/Java/Go 三端的全链路定位。接口调用应统一透传 `X-Trace-ID`（兼容 `traceparent`）。

- **Redis 弹性机制虽然完善但缺少指标暴露**：`redis_fallback.py` 实现了降级、重试、熔断器三重保护，但熔断器状态变更（CLOSED → OPEN → HALF_OPEN）和降级触发次数**没有暴露到 Prometheus 指标**，运维无法感知 Redis 健康状态的真实波动。

- **多 Worker 模式下的状态共享问题**：生产环境使用 `uvicorn --workers 4` 启动多进程，但 `TaskTracker` 和 `WebSocketService` 均为**进程内单例**（Python 全局变量），多 Worker 间无法共享任务追踪状态和 WebSocket 连接池。需要引入 Redis Pub/Sub 或共享内存机制解决跨进程协调。

- **安全防护仍偏"应用内"而非"平台级"**：虽然有 JWT、RBAC 权限、验证码等应用层安全措施，但缺少**接口限流/防重复提交**（对比 Java 端已有 `@RateLimit` + `@RepeatSubmit`）、**IP 风险识别与黑名单策略**、以及**密码错误计数锁定**（仅在规划中）。

- **测试基础设施与生产代码存在隔离不足**：`conftest.py` 使用 SQLite 内存数据库替代 MySQL 进行测试，但 SQLAlchemy 异步方言差异（aiomysql vs aiosqlite）可能导致部分 SQL 行为不一致（如自增 ID、JSON 字段、`ON DUPLICATE KEY UPDATE` 等）。
