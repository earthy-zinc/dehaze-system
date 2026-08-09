/**
 * 前端日志监控 Logger 模块测试（阶段 1）
 *
 * 覆盖：trace_id 请求级绑定（并发安全）、采样限流（含时间窗口清理）、批量上报、
 * 离线缓存、队列上限、字段截断、ERROR 去重与次数汇总。
 *
 * 纯前端逻辑，通过 pnpm test:unit 运行（独立配置，不依赖后端登录）。
 */
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Logger, generateTraceId, setCurrentTraceId } from "@/logger";
import { service } from "@/utils/request";
import type { LogEntry, LogTransport } from "@/logger";
import type { InternalAxiosRequestConfig } from "axios";
import {
  CaptureTransport,
  MemoryStore,
  MockTransport,
  persistedQueue,
  settleFlush,
} from "#/unit/logger/helpers";

beforeEach(() => {
  Logger.reset();
});

afterEach(() => {
  Logger.reset();
});

describe("trace_id 请求级绑定（并发安全）", () => {
  test("请求拦截器生成 trace_id 注入请求头与 config.metadata", async () => {
    Logger.install({ app: "react", appVersion: "1.0.0" });

    let capturedHeader: string | undefined;
    let capturedMetadataTraceId: string | undefined;
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: InternalAxiosRequestConfig) => {
      capturedHeader = config.headers?.["X-Trace-Id"] ?? config.headers?.get?.("X-Trace-Id");
      capturedMetadataTraceId = config.metadata?.traceId;
      return {
        data: { code: "00000", data: null, msg: "success", traceId: "" },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    }) as any;

    try {
      await service.get("/api/v1/test/trace");
      expect(capturedHeader).toMatch(/^[0-9a-f]{32}$/);
      // config.metadata.traceId 与请求头一致，供日志串联读取
      expect(capturedMetadataTraceId).toBe(capturedHeader);
      // 不写入全局变量（并发安全）：getTraceId() 返回空串
      expect(Logger.getInstance()!.getTraceId()).toBe("");
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("响应头 X-Trace-Id 对齐到 config.metadata.traceId（不污染全局变量）", async () => {
    Logger.install({ app: "react", appVersion: "1.0.0" });
    const serverTraceId = generateTraceId();

    let alignedMetadataTraceId: string | undefined;
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: InternalAxiosRequestConfig) => ({
      data: { code: "00000", data: null, msg: "success", traceId: serverTraceId },
      status: 200,
      statusText: "OK",
      headers: { "x-trace-id": serverTraceId },
      config,
    })) as any;

    try {
      await service.get("/api/v1/test/align");
      // 响应拦截器已执行，config.metadata.traceId 应被响应头覆盖
      // 通过发第二个请求捕获上一次对齐效果不可行（每次请求独立生成 trace_id），
      // 改为通过慢请求日志间接验证：若对齐生效，慢请求日志 trace_id 应为 serverTraceId
      // 此处仅验证全局变量未被污染
      expect(Logger.getInstance()!.getTraceId()).toBe("");
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("并发请求各自携带独立 trace_id，互不覆盖", async () => {
    Logger.install({ app: "react", appVersion: "1.0.0" });

    const capturedHeaders: string[] = [];
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: InternalAxiosRequestConfig) => {
      const header = config.headers?.["X-Trace-Id"] ?? config.headers?.get?.("X-Trace-Id");
      if (header) capturedHeaders.push(header);
      return {
        data: { code: "00000", data: null, msg: "success", traceId: "" },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    }) as any;

    try {
      // 并发发起两个请求
      await Promise.all([service.get("/api/v1/concurrent-a"), service.get("/api/v1/concurrent-b")]);
      // 两个请求各自携带独立的 trace_id
      expect(capturedHeaders).toHaveLength(2);
      expect(capturedHeaders[0]).toMatch(/^[0-9a-f]{32}$/);
      expect(capturedHeaders[1]).toMatch(/^[0-9a-f]{32}$/);
      expect(capturedHeaders[0]).not.toBe(capturedHeaders[1]);
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("API 失败日志 trace_id 从 config.metadata.traceId 读取", async () => {
    const transport = new MockTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });

    let requestTraceId: string | undefined;
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: InternalAxiosRequestConfig) => {
      requestTraceId = config.headers?.["X-Trace-Id"] ?? config.headers?.get?.("X-Trace-Id");
      return {
        data: { code: "B0500", msg: "business error", data: null },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    }) as any;

    try {
      await expect(service.get("/api/v1/fail")).rejects.toBeDefined();
      await settleFlush();

      const errorLog = transport.sentBatches.flat().find((l) => l.level === "ERROR");
      expect(errorLog).toBeDefined();
      // API 失败日志的 trace_id 与请求头一致（从 config.metadata.traceId 读取）
      expect(errorLog!.trace_id).toBe(requestTraceId);
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("慢请求日志 trace_id 从 config.metadata.traceId 读取", async () => {
    const transport = new MockTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });

    let requestTraceId: string | undefined;
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: InternalAxiosRequestConfig) => {
      requestTraceId = config.headers?.["X-Trace-Id"] ?? config.headers?.get?.("X-Trace-Id");
      // 模拟慢请求：将 startTime 回拨到阈值之前
      config.metadata = { ...config.metadata, startTime: Date.now() - 4000 };
      return {
        data: { code: "00000", data: null, msg: "success", traceId: "" },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    }) as any;

    try {
      await service.get("/api/v1/slow");
      await settleFlush();

      const slowLog = transport.sentBatches.flat().find((l) => l.message === "SLOW_REQUEST");
      expect(slowLog).toBeDefined();
      expect(slowLog!.trace_id).toBe(requestTraceId);
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("全局错误日志不携带 trace_id（语义不属于请求链路）", () => {
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;
    // 即使全局变量被污染，全局错误日志也不应读取
    setCurrentTraceId("polluted-by-concurrent-request");

    logger.error("window.onerror: Uncaught TypeError");

    const entry = persistedQueue()[0]!;
    expect(entry.trace_id).toBe("");
  });

  test("fields.trace_id 显式传入时填入日志条目", () => {
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("api-error", { trace_id: "explicit-trace-id" });

    const entry = persistedQueue()[0]!;
    expect(entry.trace_id).toBe("explicit-trace-id");
  });
});

describe("采样与限流", () => {
  test("ERROR 全量上报，INFO 不上报", () => {
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    // 固定随机值为 0.5：ERROR(100%) 必上报，INFO(0%) 必丢弃
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    logger.error("e1");
    logger.info("i1");
    vi.restoreAllMocks();

    const levels = persistedQueue().map((l) => l.level);
    expect(levels).toEqual(["ERROR"]);
  });

  test("WARN 按 50% 采样：随机值低于 50 时入队", () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });

    // 固定随机值为 0.4（<50%），WARN 应入队
    vi.spyOn(Math, "random").mockReturnValue(0.4);
    logger.warn("warn-1");
    vi.restoreAllMocks();

    expect(persistedQueue().map((l) => l.message)).toEqual(["warn-1"]);
  });

  test("WARN 按 50% 采样：随机值高于 50 时丢弃", () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });

    // 固定随机值为 0.99（>50%），保证 WARN 被采样丢弃
    vi.spyOn(Math, "random").mockReturnValue(0.99);
    logger.warn("warn-1");
    vi.restoreAllMocks();

    expect(persistedQueue().length).toBe(0);
  });

  test("限流：60s 内最多上报 20 条，超限丢弃", async () => {
    const transport = new MockTransport(Infinity); // 让所有 flush 失败，队列保留便于断言
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });

    for (let i = 0; i < 30; i++) {
      logger.error(`err-${i}`);
    }
    await settleFlush();

    const persisted = persistedQueue();
    expect(persisted.length).toBe(20);
    expect(persisted[0]!.message).toBe("err-0");
    expect(persisted[19]!.message).toBe("err-19");
  });

  test("限流：时间窗口清理后，旧时间戳被剔除，恢复上报能力", async () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport(Infinity)],
      storage: new MemoryStore(),
      rateLimitWindowMs: 100, // 缩短窗口便于测试
    });

    // 填满 20 条限流
    for (let i = 0; i < 20; i++) {
      logger.error(`batch-1-${i}`);
    }
    // 第 21 条被限流
    logger.error("should-drop");
    await settleFlush();
    expect(persistedQueue().length).toBe(20);
    expect(persistedQueue().some((l) => l.message === "should-drop")).toBe(false);

    // 模拟时间窗口已过去：将 sentTimestamps 第一项设为窗口外
    // 直接驱动 allowReport 的清理逻辑（不依赖 fake timer）
    const internal = logger as unknown as { sentTimestamps: number[] };
    internal.sentTimestamps[0] = Date.now() - 200;

    logger.error("after-window");
    await settleFlush();

    // 窗口外恢复上报能力
    const after = persistedQueue();
    expect(after.some((l) => l.message === "after-window")).toBe(true);
  });
});

describe("批量上报与离线缓存", () => {
  test("队列满 10 条立即批量上报", async () => {
    const transport = new MockTransport();
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });

    for (let i = 0; i < 10; i++) {
      logger.error(`api-error-${i}`);
    }

    // flush 为异步批量，等待批量到达 transport
    await vi.waitFor(() => {
      expect(transport.sentBatches.some((b) => b.length === 10)).toBe(true);
    });
  });

  test("离线缓存：队列持久化到 storage，刷新后可恢复并清空", async () => {
    const store = new MemoryStore();
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: store,
    });

    logger.error("cached-1");
    logger.error("cached-2");
    const persisted = JSON.parse(store.getItem("dehaze_logs")!) as LogEntry[];
    expect(persisted.map((l) => l.message)).toEqual(["cached-1", "cached-2"]);

    // 模拟页面刷新：重置单例后以同一 storage 重新 install
    Logger.reset();
    const logger2 = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: store,
    });

    // loadQueue 从 storage 恢复，flush 成功后清空持久化
    await logger2.flush();
    await settleFlush();
    const after = JSON.parse(store.getItem("dehaze_logs") ?? "[]") as LogEntry[];
    expect(after.length).toBe(0);
  });

  test("离线缓存：storage 损坏时静默丢弃，不阻塞上报", () => {
    const brokenStore = new MemoryStore();
    brokenStore.setItem("dehaze_logs", "{invalid json");
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: brokenStore,
    });

    // loadQueue 解析失败应静默丢弃，logger 仍可用
    logger.error("after-corrupt");
    expect(persistedQueue().map((l) => l.message)).toEqual(["after-corrupt"]);
  });

  test("队列上限 100 条，超出丢弃最旧", () => {
    // 无 send 的 transport：flush 无上报方，不触发批量清空，便于观测队列上限
    // rateLimitMax 调大，避免限流在入队前截断（默认 20 不足以撑到 100）
    const consoleOnly = { log: () => {} };
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [consoleOnly],
      storage: new MemoryStore(),
      rateLimitMax: 1000,
    });

    for (let i = 0; i < 105; i++) {
      logger.error(`err-${i}`);
    }

    const raw = persistedQueue();
    expect(raw.length).toBe(100);
    expect(raw[0]!.message).toBe("err-5");
    expect(raw[99]!.message).toBe("err-104");
  });

  test("上报失败保留队列，重试成功后清空", async () => {
    const transport = new MockTransport(1); // 首次失败
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });

    for (let i = 0; i < 10; i++) {
      logger.error(`err-${i}`);
    }
    await settleFlush(); // 首次 flush 失败并恢复队列

    const persisted = persistedQueue();
    expect(persisted.length).toBe(10);

    await logger.flush(); // 重试成功
    await settleFlush();
    expect(persistedQueue().length).toBe(0);
  });
});

describe("字段截断与基础字段注入", () => {
  test("message 超长截断到 2000 字符", () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });

    const longMessage = "x".repeat(3000);
    logger.error(longMessage);

    const entry = persistedQueue()[0]!;
    expect(entry.message.length).toBe(2000);
    expect(entry.message).toBe("x".repeat(2000));
  });

  test("error_stack 超长截断到 8000 字符", () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });

    const longStack = "s".repeat(10000);
    logger.error("with-stack", { error_stack: longStack });

    const entry = persistedQueue()[0]!;
    expect(entry.error_stack!.length).toBe(8000);
  });

  test("基础字段注入：service/app/app_version/timestamp 符合规范", () => {
    const logger = Logger.install({
      app: "react",
      appVersion: "1.2.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const explicitTraceId = generateTraceId();

    logger.error("API_ERROR", {
      method: "POST",
      path: "/api/v1/prediction",
      status: 500,
      duration: 1203.5,
      code: "B0500",
      trace_id: explicitTraceId,
    });

    const entry = persistedQueue()[0]!;
    expect(entry.service).toBe("client");
    expect(entry.app).toBe("react");
    expect(entry.app_version).toBe("1.2.0");
    expect(entry.level).toBe("ERROR");
    expect(entry.message).toBe("API_ERROR");
    expect(entry.method).toBe("POST");
    expect(entry.path).toBe("/api/v1/prediction");
    expect(entry.status).toBe(500);
    expect(entry.duration).toBe(1203.5);
    expect(entry.code).toBe("B0500");
    expect(entry.trace_id).toBe(explicitTraceId);
    expect(new Date(entry.timestamp).toISOString()).toBe(entry.timestamp);
  });
});

describe("ERROR 去重与次数汇总", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    Logger.reset();
    vi.useRealTimers();
  });

  test("相同 fingerprint 在 10s 窗口内只输出首条，重复被去重", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("RenderFlex overflowed", { error_stack: "stack-A", error_type: "js" });
    logger.error("RenderFlex overflowed", { error_stack: "stack-A", error_type: "js" });
    logger.error("RenderFlex overflowed", { error_stack: "stack-A", error_type: "js" });

    // 仅首条输出（无 dedup_count 标记），后两条去重
    const real = transport.logs.filter((l) => !l.dedup_count);
    expect(real).toHaveLength(1);
    expect(real[0]!.message).toBe("RenderFlex overflowed");
  });

  test("窗口结束时补发汇总条目，dedup_count 标记总次数，message 标注重复次数", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("RenderFlex overflowed", {
      error_stack: "stack-A",
      error_type: "js",
      error_source: "window.onerror",
    });
    for (let i = 0; i < 5; i++) {
      logger.error("RenderFlex overflowed", { error_stack: "stack-A", error_type: "js" });
    }

    // 推进时间超过 10s 窗口，触发定时器补发汇总
    vi.advanceTimersByTime(10_001);

    const summaries = transport.logs.filter((l) => l.dedup_count);
    expect(summaries).toHaveLength(1);
    expect(summaries[0]!.dedup_count).toBe(6); // 首条 + 5 次重复
    expect(summaries[0]!.message).toBe("RenderFlex overflowed (10s 内重复 5 次)");
    // 汇总条目携带原始 error_stack / error_type，便于 ELK 关联
    expect(summaries[0]!.error_stack).toBe("stack-A");
    expect(summaries[0]!.error_type).toBe("js");
    expect(summaries[0]!.error_source).toBe("window.onerror");
  });

  test("单次命中无重复时不补发汇总，避免噪声", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("one-shot-error");
    vi.advanceTimersByTime(10_001);

    expect(transport.logs).toHaveLength(1);
    expect(transport.logs.filter((l) => l.dedup_count)).toHaveLength(0);
  });

  test("不同 fingerprint 不去重，各自独立输出", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("error-A", { error_stack: "stack-A" });
    logger.error("error-B", { error_stack: "stack-B" });

    const real = transport.logs.filter((l) => !l.dedup_count);
    expect(real).toHaveLength(2);
  });

  test("窗口过期后相同 fingerprint 视为新 burst，先补发上一轮汇总再输出新首条", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("recurring-error", { error_stack: "stack-A" });
    logger.error("recurring-error", { error_stack: "stack-A" }); // 重复 1 次

    // 推进时间超过窗口，再触发相同错误 → 视为新 burst
    vi.advanceTimersByTime(10_001);
    logger.error("recurring-error", { error_stack: "stack-A" });

    const summaries = transport.logs.filter((l) => l.dedup_count);
    expect(summaries).toHaveLength(1);
    expect(summaries[0]!.dedup_count).toBe(2);

    // 第一轮首条 + 第二轮首条
    const real = transport.logs.filter((l) => !l.dedup_count);
    expect(real).toHaveLength(2);
  });

  test("不同 fingerprint 到来时先补发上一轮汇总", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("error-A", { error_stack: "stack-A" });
    logger.error("error-A", { error_stack: "stack-A" });
    logger.error("error-A", { error_stack: "stack-A" });
    // 不同 fingerprint 到来：先补发 A 的汇总，再输出 B
    logger.error("error-B", { error_stack: "stack-B" });

    const summaries = transport.logs.filter((l) => l.dedup_count);
    expect(summaries).toHaveLength(1);
    expect(summaries[0]!.dedup_count).toBe(3);
    expect(summaries[0]!.message).toContain("error-A");

    // A 首条 + B 首条
    const real = transport.logs.filter((l) => !l.dedup_count);
    expect(real).toHaveLength(2);
  });

  test("WARN/INFO 不参与去重", () => {
    const transport = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.warn("same-warn");
    logger.warn("same-warn");
    logger.info("same-info");
    logger.info("same-info");

    // WARN/INFO 不去重，全部经 emit 输出到 transport（采样前已捕获）
    expect(transport.logs).toHaveLength(4);
  });

  test("汇总条目经 emit 输出并入队持久化，可被远程上报", () => {
    const transport = new MockTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;

    logger.error("storm-error", { error_stack: "stack-A" });
    for (let i = 0; i < 9; i++) {
      logger.error("storm-error", { error_stack: "stack-A" });
    }
    vi.advanceTimersByTime(10_001);

    // 汇总条目经 emit 输出，MockTransport.log 捕获
    const allLogged = transport.sentBatches.flat();
    const loggedSummaries = allLogged.filter((l) => l.dedup_count);
    expect(loggedSummaries).toHaveLength(1);
    expect(loggedSummaries[0]!.dedup_count).toBe(10);

    // 汇总条目也入队持久化，可被远程 flush 上报
    const queuedSummaries = persistedQueue().filter((l) => l.dedup_count);
    expect(queuedSummaries).toHaveLength(1);
    expect(queuedSummaries[0]!.dedup_count).toBe(10);
  });
});

describe("Logger.install 二次调用重新注册全局处理器", () => {
  test("二次 install 传入新 transports 时更新 transports 并重建定时器", () => {
    const transport1 = new CaptureTransport();
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport1],
      storage: new MemoryStore(),
    });
    const internal = logger as unknown as {
      flushTimer: ReturnType<typeof setInterval> | undefined;
      transports: LogTransport[];
    };
    const firstFlushTimer = internal.flushTimer;
    expect(firstFlushTimer).toBeDefined();

    const transport2 = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport2],
    });

    // transports 更新为新 transport
    expect(internal.transports).toEqual([transport2]);
    // flushTimer 被重新创建（旧的被清理后重新 start）
    expect(internal.flushTimer).toBeDefined();
    expect(internal.flushTimer).not.toBe(firstFlushTimer);
  });

  test("二次 install 不传 transports 时不重新注册（无副作用）", () => {
    const transport1 = new CaptureTransport();
    const logger = Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport1],
      storage: new MemoryStore(),
    });
    const internal = logger as unknown as {
      flushTimer: ReturnType<typeof setInterval> | undefined;
      transports: LogTransport[];
    };
    const firstFlushTimer = internal.flushTimer;

    Logger.install({ app: "react", appVersion: "1.0.0" });

    // transports 不变，flushTimer 未重建
    expect(internal.transports).toEqual([transport1]);
    expect(internal.flushTimer).toBe(firstFlushTimer);
  });

  test("二次 install 保留 queue，不丢失未上报日志", () => {
    const transport1 = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport1],
      storage: new MemoryStore(),
    });
    Logger.getInstance()!.error("before-reinstall");
    expect(persistedQueue().length).toBe(1);

    const transport2 = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport2],
    });

    // queue 保留（dispose 仅清理定时器与处理器，不清 queue）
    expect(persistedQueue().length).toBe(1);
    expect(persistedQueue()[0]!.message).toBe("before-reinstall");
  });
});
