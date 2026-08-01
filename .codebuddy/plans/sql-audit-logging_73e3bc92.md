---
name: sql-audit-logging
overview: 为 dehaze-python/dehaze-go/dehaze-java 三端统一开启 SQL 审计日志，设计统一的 JSON 结构化字段 schema，与现有日志架构对齐，支持按环境配置开关和级别。
todos:
  - id: py-config
    content: Python：config.py 新增 SQL_LOG_LEVEL 字段，基类默认 INFO，ProductionSettings 覆写为 WARN
    status: completed
  - id: py-database
    content: Python：database.py 注册 engine.sync_engine 的 before_cursor_execute / after_cursor_execute 事件监听器，以 logging.getLogger("sql") 输出结构化 SQL 日志；移除 echo 参数依赖
    status: completed
    dependencies:
      - py-config
  - id: go-logger
    content: Go：重构 GormLogger.Trace()——正常 SQL 从 Debug 提升至 Info，慢查询保持 Warn，错误保持 Error；统一附加 zap.String("logger","sql") 及结构化字段
    status: completed
  - id: java-interceptor
    content: Java：新建 SqlLogInterceptor（MyBatis Interceptor），拦截 StatementHandler 捕获 SQL，以 StructuredArguments 输出 INFO 日志；在 MybatisConfig 中注册
    status: completed
  - id: java-config
    content: Java：application-dev.yml / application-test.yml 新增 dehaze.sql-log.level=INFO 属性
    status: completed
    dependencies:
      - java-interceptor
  - id: doc-update
    content: 文档：更新 07-日志架构设计.md，新增 SQL 审计日志章节（JSON 字段规范、环境差异化、使用方式）
    status: completed
    dependencies:
      - py-database
      - go-logger
      - java-interceptor
---

## 用户需求

为 dehaze-python、dehaze-go、dehaze-java 三端后端开启 SQL 审计日志。日志格式必须对齐 `dehaze-doc/docs/02-系统架构/07-日志架构设计.md` 中定义的 NDJSON 结构化日志规范。先输出实现方案。

## 核心功能

- **Python**：用 SQLAlchemy 事件监听器替代原始 `echo` 参数，将 SQL 以结构化 JSON 输出至 info.log
- **Go**：改造 `GormLogger.Trace()`，将正常 SQL 从 DEBUG 提升至 INFO 级别，使用结构化字段
- **Java**：新建 MyBatis Interceptor 拦截 SQL 执行，以 INFO 级别输出结构化日志
- 不新增 sql.log 文件，统一以 `logger: "sql"` 标识区分
- 三种消息类型：SQL（INFO）、SLOW_SQL（WARN）、SQL_ERROR（ERROR）
- 场景字段：`sql`、`duration_ms`、`rows`、`threshold_ms`（慢查询时）、`error`（错误时）
- 请求上下文（`trace_id`、`method`、`path`、`ip`、`user_agent`、`user_id`）自动注入
- 按环境控制：dev/test 默认 INFO（全量 SQL），prod 默认 WARN（仅慢查询+错误）

## 技术方案

### 架构原则

- **理由：架构文档明确规定"DEBUG 级别不落盘"**，因此正常 SQL 必须使用 INFO 级别以确保写入 info.log
- **理由：不在 message 中内联机器可读键值**（如 `sql=... rows=...`），必须用独立 JSON 字段

### Python 方案 — SQLAlchemy 事件监听器替代 echo

当前 `echo=True` 将原始 SQL 打印到 stderr，完全不经过 Python logging 管线，无法获得结构化字段和请求上下文注入。改为注册 `engine.sync_engine` 的 `before_cursor_execute` / `after_cursor_execute` 事件监听器：

1. `before_cursor_execute`：记录开始时间戳到 `_sql_timers` 字典（key=connection id）
2. `after_cursor_execute`：计算耗时 `duration_ms`，以 `logging.getLogger("sql").info("SQL", extra={...})` 输出
3. `extra` 中的字段由 `pythonjsonlogger.JsonFormatter` 自动提升为顶层 JSON key
4. 新增 `SQL_LOG_LEVEL` 配置项：`"INFO"`（全量 SQL）或 `"WARN"`（仅慢查询+错误）

### Go 方案 — GormLogger.Trace() 级别提升

当前 `Trace()` 将正常 SQL 以 `l.log(ctx).Debug(...)` 输出，但 Zap 文件 core 仅接受 INFO+ 级别，SQL 从未落盘。修改方案：

1. 正常 SQL 分支（`l.LogLevel >= gormLogger.Info`）改为 `l.log(ctx).Info(...)`
2. 结构化字段：`zap.String("sql", sql)`、`zap.Int64("rows", rows)`、`zap.Float64("duration_ms", ...)`、`zap.String("logger", "sql")`
3. 慢查询保持 WARN 级别，SQL 错误保持 ERROR 级别
4. `logger.WithContext(ctx)` 自动注入 `trace_id` 等请求上下文字段
5. 现有 `log-mode: info` 配置不变，仅改变输出级别

### Java 方案 — MyBatis Interceptor

当前 `log-impl` 被注释，MyBatis 默认通过 Slf4jImpl 输出 DEBUG（不落盘）。新建 `SqlLogInterceptor`（实现 `org.apache.ibatis.plugin.Interceptor`）：

1. 拦截 `StatementHandler` 的 `query` / `update` / `batch` 方法（`@Intercepts({...})`）
2. 通过 `StatementHandler.getBoundSql()` 提取 SQL 元信息
3. 计算耗时，构造结构化日志
4. 使用 `net.logstash.logback.argument.StructuredArguments` 将字段注入为 JSON 顶层 key
5. Logger 名称为 `"sql"`，由 Logback `LogstashEncoder` 自动注入 MDC 请求上下文
6. 新增 `dehaze.sql-log.level` 属性控制输出级别

### SQL 日志 JSON 字段规范

```
基础字段（继承自日志架构 §3）：
  timestamp, level, logger("sql"), service, thread, message
请求上下文（自动注入）：
  trace_id, method, path, ip, user_agent, user_id

场景字段（本次新增）：
  sql          — SQL 语句字符串
  duration_ms  — 执行耗时（毫秒）
  rows         — 影响/返回行数
  threshold_ms — 慢查询阈值，仅 SLOW_SQL(WARN) 时出现
  error        — 错误信息，仅 SQL_ERROR(ERROR) 时出现
```

### 环境差异化配置

| 环境 | Python `SQL_LOG_LEVEL` | Go `log-mode` | Java `dehaze.sql-log.level` | 输出内容 |
| --- | --- | --- | --- | --- |
| dev | INFO | info（不变） | INFO | 全量 SQL + 慢查询 + 错误 |
| test | INFO | info（不变） | INFO | 全量 SQL + 慢查询 + 错误 |
| prod | WARN | warn（待建 config.prod.yaml） | WARN | 仅慢查询 + 错误 |


## Agent Extensions

### SubAgent

- **code agent**
- Purpose：执行三端的 SQL 审计日志代码实现，包括 Python 事件监听器、Go GormLogger 改造、Java MyBatis Interceptor 新建
- Expected outcome：三端 SQL 审计日志正确输出至 NDJSON 格式 info.log，字段对齐架构规范

- **code-explorer**
- Purpose：探索三端现有日志管线中隐蔽的依赖点（如 Python 的 JsonFormatter.add_fields 如何传递 extra、Java 的 LogstashEncoder 如何输出 StructuredArguments）
- Expected outcome：确认所有实现细节路径准确无误，避免字段丢失或格式错误