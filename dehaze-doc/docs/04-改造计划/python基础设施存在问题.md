### 现有 Python 基础设施/架构主要问题（基于当前代码与文档状态）

> 更新说明（2026-07-12）：已对各项问题进行代码层面的实际核对，标注当前真实状态。

- **操作日志中间件已迁移到纯 ASGI 实现**（✅ 已修复）：`OperationLogMiddleware` 已从 `BaseHTTPMiddleware` 迁移到纯 ASGI 中间件（见 `dehaze-python/app/middleware/operation_log.py`）。新实现通过 tee receive/send 流式采集请求体和响应体，不会将整个请求体读入内存，正确处理 StreamingResponse（基于响应头检测，不缓冲完整响应体），且不再创建 Request/Response 包装对象。同时修复了 user_id 获取方式：从 contextvar `get_current_user_id()` 获取（由 auth 依赖注入设置），替代原来始终返回 None 的 `request.state.user_id`。

- **链路可观测性部分实现，但跨端全链路追踪仍缺失**（部分实现）：Python 端已有 `TraceMiddleware`（`dehaze-python/app/middleware/trace.py`）支持 `X-Trace-Id` / `X-Request-Id` 透传与回写，Go 端 `pkg/trace/trace.go` 也已支持 W3C `traceparent` 与 SkyWalking `sw8` 头解析。但**仍未接入 OpenTelemetry / Jaeger / Zipkin** 等真正的分布式追踪后端，无法形成跨 Python/Java/Go 三端的可视化全链路。下一步需引入 OpenTelemetry SDK 并对接追踪后端。

- **Redis 弹性机制虽然完善但缺少指标暴露**（仍存在）：`redis_fallback.py` 实现了降级、重试、熔断器三重保护，但熔断器状态变更（CLOSED → OPEN → HALF_OPEN）和降级触发次数**没有暴露到 Prometheus 指标**，运维无法感知 Redis 健康状态的真实波动。

- **多 Worker 模式下的状态共享问题已修复**（✅ 已修复）：
  - **WebSocketService** 已从进程内单例升级为 `DistributedConnectionManager`（见 `dehaze-python/app/service/websocket_service.py`），通过 Redis Pub/Sub 实现跨 Worker 消息广播。每个 Worker 维护本地连接，`send_personal` / `broadcast` 通过 Redis 频道 `dehaze:ws:broadcast` 跨 Worker 投递。在线用户列表通过 Redis sorted set + 心跳维护。Redis 不可用时自动降级为本地单 Worker 模式。
  - **TaskTracker** 已增加 Redis 背景状态同步（见 `dehaze-python/app/service/task_tracker.py`）。本地追踪仍用于优雅关闭（各 Worker 只等待自己的任务），Redis hash 用于全局视图（`get_global_running_tasks()` / `get_global_running_count()`），心跳定期续期 TTL。

- **安全防护已从"应用内"升级为"平台级"**（✅ 已修复）：
  - **接口限流**：新增 `rate_limit` 依赖（`dehaze-python/app/decorators/rate_limit.py`），基于 Redis sorted set 滑动窗口算法，支持按 user_id 或 IP 限流。用法：`@router.post(..., dependencies=[Depends(rate_limit(times=10, seconds=60))])`
  - **防重复提交**：新增 `repeat_submit` 依赖（`dehaze-python/app/decorators/repeat_submit.py`），基于 Redis SETNX + TTL，以 (用户ID|IP + 路径 + 请求体哈希) 为 key 防重复。用法：`@router.post(..., dependencies=[Depends(repeat_submit(interval=5))])`
  - **IP 黑名单**：新增 `IPBlacklistMiddleware` 纯 ASGI 中间件（`dehaze-python/app/middleware/ip_blacklist.py`），自动追踪异常请求（4xx/5xx），超过阈值自动封禁 IP。同时提供 `IPBlacklistService` 供管理端手动管理黑名单。
  - **密码错误计数锁定**：已存在于 `dehaze-python/app/router/auth.py`（5 次失败后锁定 15 分钟），无需重复实现。
  - 所有安全组件均支持 fail-open 降级：Redis 不可用时放行请求，避免安全组件故障导致服务不可用。

- **测试基础设施与生产代码存在隔离不足**（仍存在）：`conftest.py` 使用 SQLite 内存数据库替代 MySQL 进行测试，但 SQLAlchemy 异步方言差异（aiomysql vs aiosqlite）可能导致部分 SQL 行为不一致（如自增 ID、JSON 字段、`ON DUPLICATE KEY UPDATE` 等）。
