### 现有 Python 基础设施/架构主要问题（基于当前代码与文档状态）

> 更新说明（2026-07-12）：已对各项问题进行代码层面的实际核对，标注当前真实状态。

- **链路可观测性部分实现，但跨端全链路追踪仍缺失**（部分实现）：Python 端已有 `TraceMiddleware`（`dehaze-python/app/middleware/trace.py`）支持 `X-Trace-Id` / `X-Request-Id` 透传与回写，Go 端 `pkg/trace/trace.go` 也已支持 W3C `traceparent` 与 SkyWalking `sw8` 头解析。但**仍未接入 OpenTelemetry / Jaeger / Zipkin** 等真正的分布式追踪后端，无法形成跨 Python/Java/Go 三端的可视化全链路。下一步需引入 OpenTelemetry SDK 并对接追踪后端。

- **Redis 弹性机制虽然完善但缺少指标暴露**（仍存在）：`redis_fallback.py` 实现了降级、重试、熔断器三重保护，但熔断器状态变更（CLOSED → OPEN → HALF_OPEN）和降级触发次数**没有暴露到 Prometheus 指标**，运维无法感知 Redis 健康状态的真实波动。


- **测试基础设施与生产代码存在隔离不足**（仍存在）：`conftest.py` 使用 SQLite 内存数据库替代 MySQL 进行测试，但 SQLAlchemy 异步方言差异（aiomysql vs aiosqlite）可能导致部分 SQL 行为不一致（如自增 ID、JSON 字段、`ON DUPLICATE KEY UPDATE` 等）。
