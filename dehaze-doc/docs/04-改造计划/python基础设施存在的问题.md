# dehaze-python 后端特定问题与改进建议

> 文档定位：仅收录 **Python 特有问题**（性能/优化/bug），即由 Python 语言特性或 FastAPI/SQLAlchemy 框架特性导致的实现缺陷。
>
> 跨三端的通用问题（架构设计层面 + 模块业务设计层面）已提取到 [通用基础设施问题与改进](./通用基础设施问题与改进.md)。
>
> 核对基准日期：2026-07-13
> 审查范围：`dehaze-python` 全部基础设施层代码
> 审查方法：对照 [08-后端基础设施设计(Python)](../02-系统架构/08-后端基础设施设计(Python).md) 核对实际实现

---

## 一、问题总览

| 领域 | 严重 | 高 | 中 | 低 |
|------|------|---|---|---|
| 缓存体系 | 0 | 1 | 0 | 0 |
| 数据访问层 | 0 | 1 | 1 | 0 |
| 安全防护 | 0 | 0 | 2 | 0 |
| 部署与多 Worker | 0 | 1 | 1 | 0 |
| 日志与可观测性 | 0 | 0 | 2 | 0 |

> 通用问题（TraceId 跨端追踪、缓存防护体系、异步任务可靠性、熔断/重试、数据权限异步失效、审计字段异步填充等）见 [通用基础设施问题与改进](./通用基础设施问题与改进.md)

---

## 二、缓存体系（Python 特有）

### 2.1 [HIGH] RedisCircuitBreaker 与 with_redis_retry 为死代码

[app/infrastructure/cache/redis_fallback.py](file:///e:/DehazeSystem/dehaze-python/app/infrastructure/cache/redis_fallback.py) 定义了 `RedisCircuitBreaker` 熔断器和 `with_redis_retry` 重试装饰器，但**全代码库零调用**。当前仅使用 `redis_operation_with_fallback()` 降级函数，熔断器状态变更（CLOSED → OPEN → HALF_OPEN）和降级触发次数没有暴露到 Prometheus 指标，运维无法感知 Redis 健康状态的真实波动。

**改进建议**：接入熔断器/重试，并将熔断器状态与降级次数暴露为 Prometheus 指标。

---

## 三、数据访问层（Python 特有）

### 3.1 [HIGH] Service 层存在多处显式 db.commit()，违反事务边界设计

[08-后端基础设施设计(Python) §6.3](file:///e:/DehazeSystem/dehaze-doc/docs/02-系统架构/08-后端基础设施设计(Python).md) 明确设计规范：Router 层持有事务边界（`Depends(get_db)` 请求结束自动 commit/rollback），Service 层**不做 commit**，Repository 层只做 `flush()`。

实际代码中 Service 层存在多处显式 `db.commit()`，以及少数 Router 层显式 commit（为解决 `get_db()` yield 后置 commit 跨请求不可见问题）。

**改进建议**：清理 Service 层的显式 `db.commit()`，遵循"请求边界 = 事务边界"设计；若需多事务编排，使用 `get_db_session()` 上下文管理器显式管理。

### 3.2 [MEDIUM] before_update 事件对 Core 层批量更新不触发，审计字段丢失

[app/models/base.py](file:///e:/DehazeSystem/dehaze-python/app/models/base.py) 通过 SQLAlchemy `event.listens_for` 实现审计字段自动填充，但 `before_update` 事件对 Core 层批量更新（`update().where(...)`）**不触发**，批量软删除的审计字段会丢失。

**改进建议**：批量更新操作显式设置审计字段。

---

## 四、安全防护（Python 特有）

### 4.1 [MEDIUM] Token 刷新未失效旧 Token（可重放）

[app/router/auth.py](file:///e:/DehazeSystem/dehaze-python/app/router/auth.py) 的 `refresh` 接口签发新 Token 后未将旧 Token 的 jti 加入黑名单，旧 Token 在过期前仍可使用。

**改进建议**：刷新时将旧 Access Token 的 jti 写入 Redis 黑名单（TTL = 剩余有效期）。

### 4.2 [MEDIUM] fnmatch 权限匹配跨平台大小写不一致

[app/decorators/permission.py](file:///e:/DehazeSystem/dehaze-python/app/decorators/permission.py) 使用 `fnmatch.fnmatch` 进行权限通配符匹配，该函数在 Windows 上大小写不敏感、Linux 上大小写敏感。应改用 `fnmatch.fnmatchcase` 保证跨平台一致。

---

## 五、部署与多 Worker（Python 特有）

### 5.1 [HIGH] uvicorn 多 Worker 下 XXL-Job executor 端口冲突

[app/infrastructure/job/executor.py](file:///e:/DehazeSystem/dehaze-python/app/infrastructure/job/executor.py) 使用 `PyxxlRunner.run_with_daemon()` 启动 daemon 子进程监听 9999 端口。uvicorn 多 Worker 部署下，**每个 Worker 都会启动一个 daemon 监听同一端口**，第二个 Worker 起会因端口占用而失败。

**改进建议**：使用文件锁或环境变量标记确保仅首个 Worker 启动 daemon；或改用外部进程管理（supervisor/systemd）独立运行 pyxxl。

### 5.2 [MEDIUM] Prometheus 多 Worker 指标未聚合

uvicorn 多 Worker 部署下未配置 `PROMETHEUS_MULTIPROC_DIR`，每个 Worker 进程独立维护指标，`/metrics` 端点仅返回命中 Worker 的局部数据，而非全局聚合。

**改进建议**：配置 `PROMETHEUS_MULTIPROC_DIR` 环境变量 + 使用 `MultiProcessCollector` 聚合多 Worker 指标。

---

## 六、日志与可观测性（Python 特有）

### 6.1 [MEDIUM] 推理指标已定义但未接入 prediction_service

[app/infrastructure/metrics/inference_metrics.py](file:///e:/DehazeSystem/dehaze-python/app/infrastructure/metrics/inference_metrics.py) 定义了推理耗时 Histogram、推理请求 Counter 等指标，但 `prediction_service.predict` 方法**未接入** `@track_inference` 装饰器，指标全部为空。

### 6.2 [MEDIUM] f-string 日志未使用懒求值

全项目大量 `logger.info(f"...")` 使用 f-string 直接格式化字符串，即使日志级别低于 INFO 也会执行字符串拼接。应改为 `logger.info("...", arg)` 位置参数方式，利用 logging 的懒求值优化。

---
---

## 八、修复优先级清单

### P1（重要）

| # | 问题 | 文件 |
|---|------|------|
| 1 | RedisCircuitBreaker/with_redis_retry 死代码 | redis_fallback.py |
| 2 | Service 层显式 db.commit() 违反设计 | service 层 |
| 3 | XXL-Job 多 Worker 端口冲突 | executor.py |

### P2（改进）

| # | 问题 | 文件 |
|---|------|------|
| 4 | before_update 批量更新审计字段丢失 | base.py |
| 5 | Token 刷新未失效旧 Token | auth.py |
| 6 | fnmatch 跨平台大小写不一致 | permission.py |
| 8 | Prometheus 多 Worker 指标未聚合 | 部署配置 |
| 9 | 推理指标未接入 | prediction_service.py |
| 10 | f-string 日志懒求值 | 全项目 |
