# JS SDK 架构改造计划

> 本文档聚焦 `dehaze-sdk-js` 在**代码架构层面**的实际问题与改造方向，供后续重构参考。文档失真问题已在 [08-SDK架构文档.md](../04-项目实现/前端/08-SDK架构文档.md) 修复中处理，本文不重复。

## 一、问题总览

| # | 问题 | 类别 | 优先级 | 依据 |
|---|------|------|:------:|------|
| 1 | ERROR 去重（fingerprint 10s 窗口）文档声称已实现但代码缺失 | 文档与代码不一致 | P0 | [07-日志架构设计.md §3.5.4](../02-系统架构/07-日志架构设计.md)、[08-SDK架构文档.md §5.2](../04-项目实现/前端/08-SDK架构文档.md) |
| 2 | trace_id 用模块级全局变量管理，并发请求场景下串号 | 可靠性 | P1 | [Logger.ts:439-464](file:///e:/DehazeSystem/dehaze-sdk-js/src/logger/Logger.ts) |
| 3 | `Logger.install` 二次调用不重新注册全局处理器 | 正确性 | P1 | [Logger.ts:99-113](file:///e:/DehazeSystem/dehaze-sdk-js/src/logger/Logger.ts) |
| 4 | 手写 API 模块未为 OpenAPI 自动生成预留目录边界 | 演进规划 | P2 | [04-API规范.md §9.2](../02-系统架构/04-API规范.md) |

---

## 二、P0：ERROR 去重功能缺失

### 2.1 现状

日志架构设计文档 §3.5.4 明确声明：

> **ERROR 级别去重**：相同 `message + error_stack` fingerprint 在 10s 窗口内只记录一次（含 transports 输出、本地文件、ELK 上报三层），防止布局/渲染错误在每帧重绘时反复触发日志风暴。

SDK 架构文档 §5.2 设计原则表也列出了该原则。但在 [Logger.ts](file:///e:/DehazeSystem/dehaze-sdk-js/src/logger/Logger.ts) 的 `log()` 方法中（行 149-184），采样过滤与限流逻辑均存在，但**没有任何 fingerprint 计算与去重窗口逻辑**（全局搜索 `fingerprint`/`dedupe`/`去重` 零匹配）。

**三端对照**：Flutter 端已实现（`logger.dart` 用单变量 `_lastErrorFingerprint:int?` + `_lastErrorTime:DateTime?`，10s 窗口命中跳过），Android 端未实现。JS 端需与 Flutter 端对齐。

### 2.2 影响

- **文档误导**：排查者依据文档认为错误日志已被去重，实际未去重，导致对日志量的预期与实际不符
- **日志风暴风险真实存在**：`window.onerror` 捕获的渲染错误、RN `ErrorUtils.setGlobalHandler` 捕获的布局错误（如 `RenderFlex overflowed`）在 layout 阶段可高频触发，60s 内 20 条限流虽能兜底，但本地 console 输出（`ConsoleTransport` 不受采样/限流影响）会刷屏
- **三端行为不一致**：Flutter 已实现，JS 与 Android 未实现

### 2.3 改造方向

**参考 Flutter 端单变量方案补实现**（不使用 Map，避免过度设计）：

在 `Logger.log()` 采样过滤之前、transport 输出之前插入去重逻辑：

- 新增两个实例字段：`lastErrorFingerprint: string` + `lastErrorTime: number`（timestamp）
- fingerprint = `message + error_stack` 的简单 hash（无需强 hash，字符串长度 + 字符累加即可）
- 窗口 10s，命中（fingerprint 相同且 `now - lastErrorTime < 10_000`）则直接 return，跳过 transport 输出与入队上报
- 不同 `trace_id` 的相同错误仍去重（文档明确"不同 trace_id / 不同 fingerprint 的错误不受影响"指的是不同 fingerprint 不受影响，相同 fingerprint 不同 trace_id 仍去重）
- 仅对 ERROR 级别生效，WARN/INFO 不去重（采样率已足够低）

---

## 三、P1：trace_id 并发安全缺陷

### 3.1 现状

trace_id 的管理完全依赖模块级全局变量（[Logger.ts:439-464](file:///e:/DehazeSystem/dehaze-sdk-js/src/logger/Logger.ts)）：

```typescript
let currentTraceId = "";  // 模块级全局

export function getCurrentTraceId(): string { return currentTraceId; }
export function setCurrentTraceId(traceId: string): void { currentTraceId = traceId; }
```

请求拦截器调用 `ensureTraceId()` 写入全局变量，响应拦截器调用 `alignTraceId()` 覆盖全局变量，`log()` 通过 `getCurrentTraceId()` 读取全局变量填充日志条目。

### 3.2 问题

并发请求场景下全局变量被覆盖，导致非请求相关日志的 trace_id 串号：

```
时间线：
  t1: 请求 A 拦截器 → setCurrentTraceId("trace_A") → 发出请求 A
  t2: 请求 B 拦截器 → setCurrentTraceId("trace_B") → 发出请求 B  ← 全局变量被覆盖
  t3: window.onerror 触发 → logger.error() → getCurrentTraceId() 返回 "trace_B"
      → 该错误的 trace_id 错误地关联到请求 B，而非请求 A（或无关联）
  t4: 请求 A 响应返回 → alignTraceId("trace_A") → 恢复，但 t3 的日志已上报
```

**请求相关日志路径**（已正确对齐，本次错误日志本身的 trace_id 正确）：
- `reportApiError`：在记录日志前先调用 `alignResponseTraceId(response)` 对齐，再 `logger.error()`
- `reportSlowRequest`：在响应拦截器内同步调用，此时 trace_id 已对齐

> 注：`reportApiError` 调用 `alignResponseTraceId` 会将全局变量设为该错误响应的 trace_id，对紧接着的非请求日志（如性能日志）仍有副作用，改造后此副作用随全局变量弃用而消除。

**受影响的路径**（非请求日志读全局变量，串号）：
- `window.onerror` / `unhandledrejection` / 资源 error：全局错误捕获时的 trace_id 是"最近一次请求"的，可能与触发错误的实际请求无关
- 性能采集日志（LCP/INP/CLS 等）：页面加载性能日志的 trace_id 是"最近一次 API 请求"的，语义错误（性能日志本不应携带请求级 trace_id）

### 3.3 影响

- trace_id 端到端串联在并发场景下不准确，ELK 中按 trace_id 检索时前端错误日志可能关联到错误的请求
- 性能日志携带请求级 trace_id 本身语义就不对（性能指标不是某个 API 请求的产物）
- 项目将 trace_id 端到端串联作为核心监控能力（07-日志架构设计.md §3.5.3 专门描述），此缺陷削弱了该能力的可信度

### 3.4 改造方向

**将 trace_id 从"全局变量"改为"请求上下文绑定"**：

1. **请求级 trace_id 绑定到 AxiosConfig**：
   - 请求拦截器将生成的 trace_id 写入 `config.metadata.traceId`（已有 `metadata.startTime` 的先例）
   - 响应拦截器从 `config.metadata.traceId` 读取，写入响应头对齐
   - `reportApiError` / `reportSlowRequest` 从 `error.config.metadata.traceId` 读取，显式传入 `fields.trace_id`

2. **全局错误/性能日志不携带 trace_id**：
   - `window.onerror` / `unhandledrejection` / 性能采集日志：`trace_id` 留空（字段规范中 `trace_id` 非必填）
   - 这些日志本身不属于请求链路，强行关联 trace_id 语义错误

3. **`generateTraceId` / `getCurrentTraceId` / `setCurrentTraceId` 保留导出**：
   - 供宿主项目在非 SDK 请求链路（如 WebSocket、fetch 直调）中手动管理 trace_id
   - SDK 内部不再使用全局变量填充日志条目的 `trace_id`，仅由请求上下文（`fields.trace_id`）或留空决定

---

## 四、P1：Logger.install 二次调用不重新注册全局处理器

### 4.1 现状

[Logger.ts:99-113](file:///e:/DehazeSystem/dehaze-sdk-js/src/logger/Logger.ts) 的 `install()` 方法：

```typescript
static install(config: InstallConfig): Logger {
  if (config.react !== undefined) { bindReact(config.react); }
  if (Logger.instance) {
    Logger.instance.configure(config);  // 仅更新 transports
    return Logger.instance;
  }
  const logger = new Logger(config);
  Logger.instance = logger;
  logger.registerGlobalHandlers();      // 仅首次 install 执行
  logger.startPerformanceMonitoring();
  logger.startFlushTimer();
  return logger;
}
```

二次调用 `install` 走 `configure()` 仅更新 `transports`，**不重新执行** `registerGlobalHandlers` / `startPerformanceMonitoring` / `startFlushTimer`。

### 4.2 问题

- 若二次 `install` 传入新的 transports（如开发环境切换到生产配置），全局错误捕获仍走旧 transport，日志输出到已失效的 transport
- HMR 热更新场景下，新代码模块的 `install` 调用不会重注册处理器，旧模块的处理器引用旧闭包
- `Logger.reset()` 后再次 `install` 能正常注册（因为 instance 已清空），但同实例的二次 `install` 不会

### 4.3 改造方向

二次 `install` 时，若 `transports` 发生变化，需重新注册全局处理器：

- `configure()` 方法中检测 transports 是否变化，变化时先 `dispose` 旧处理器（`registeredHandlers` removeEventListener + `disposePerformance`）再重新 `registerGlobalHandlers` + `startPerformanceMonitoring`
- 或更简单：`install` 检测到 `Logger.instance` 已存在且 `transports` 变化时，先 `reset()` 再走完整初始化流程

---

## 五、P2：手写 API 模块未为 OpenAPI 自动生成预留目录边界

### 5.1 现状

[04-API规范.md §9.2](../02-系统架构/04-API规范.md) 明确演进方向：

> **与手写 SDK 的关系**：现有手写 SDK（`dehaze-sdk-js`、`dehaze-android/sdk`）的网络层封装（拦截器、错误处理、trace_id 透传）保留为上层封装，**生成层替换手写的接口定义部分**。

当前 `dehaze-sdk-js` 的 21 个 API 模块（`src/api/*/index.ts` + `model.ts`）完全手写，与后端契约靠人工同步。API 层与网络层在代码上分离（API 模块调 `request()`），但目录结构上未区分手写层与生成层。

### 5.2 影响

- OpenAPI 生成链路落地时，需从 `src/api/` 中剥离"哪些被生成替换、哪些保留手写"，增加迁移成本
- 手写 API 模块的 model 类型与后端契约靠人工同步，后端新增/修改接口时 SDK 容易滞后

### 5.3 改造方向

**暂不重组目录，仅约束新增模块**：

- 当前阶段不立即重组目录（OpenAPI 生成链路尚未落地，提前重组增加无谓变更）
- 新增 API 模块遵循"纯接口定义、不含业务逻辑"约束，便于未来生成层直接替换
- 在 [08-SDK架构文档.md](../04-项目实现/前端/08-SDK架构文档.md) §3.1 补充该演进方向说明
- 待 OpenAPI 生成链路落地时，将 `src/api/` 重命名为 `src/generated/`，手写网络层/配置/类型重组为 `src/core/`

---

## 六、不在改造范围内的事项

以下问题经评估后认为价值有限或非架构层面，不在本计划范围：

- **configAxios 拦截器为覆盖式**：当前用法是应用启动时一次性注入，覆盖式设计可接受
- **ResultEnum 硬编码业务错误码**：SDK 常见做法，OpenAPI 生成后可解决
- **测试强依赖真实后端**：集成测试的本质决定，且已有单元测试层覆盖纯逻辑
- **API 方法命名不完全统一**（getPage / listMy / getList）：对应不同语义，非同质化问题
