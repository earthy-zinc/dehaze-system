/**
 * 前端日志监控 Logger 模块测试（阶段 1）
 *
 * 覆盖：trace_id 透传与对齐、采样限流（含时间窗口清理）、批量上报、
 * 离线缓存、队列上限、字段截断、trace_id 优先级。
 *
 * 纯前端逻辑，通过 pnpm test:unit 运行（独立配置，不依赖后端登录）。
 */
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Logger, generateTraceId, setCurrentTraceId } from "@/logger";
import { service } from "@/utils/request";
import type { LogEntry } from "@/logger";
import { MemoryStore, MockTransport, persistedQueue, settleFlush } from "#/unit/logger/helpers";

beforeEach(() => {
  Logger.reset();
});

afterEach(() => {
  Logger.reset();
});

describe("trace_id 透传（请求头注入）", () => {
  test("请求拦截器注入 X-Trace-Id 请求头", async () => {
    Logger.install({ app: "react", appVersion: "1.0.0", });

    let capturedHeader: string | undefined;
    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: any) => {
      capturedHeader = config.headers?.["X-Trace-Id"] ?? config.headers?.get?.("X-Trace-Id");
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
      expect(capturedHeader).toBe(Logger.getInstance()!.getTraceId());
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("响应头 X-Trace-Id 与本地 trace_id 对齐", async () => {
    const logger = Logger.install({ app: "react", appVersion: "1.0.0", });
    const serverTraceId = generateTraceId();

    const originalAdapter = service.defaults.adapter;
    service.defaults.adapter = (async (config: any) => ({
      data: { code: "00000", data: null, msg: "success", traceId: serverTraceId },
      status: 200,
      statusText: "OK",
      headers: { "x-trace-id": serverTraceId },
      config,
    })) as any;

    try {
      await service.get("/api/v1/test/align");
      expect(logger.getTraceId()).toBe(serverTraceId);
    } finally {
      service.defaults.adapter = originalAdapter as any;
    }
  });

  test("fields.trace_id 优先于 getCurrentTraceId（外部显式指定保留）", () => {
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;
    setCurrentTraceId("global-trace-id");

    logger.error("api-error", { trace_id: "explicit-trace-id" });

    const entry = persistedQueue()[0]!;
    expect(entry.trace_id).toBe("explicit-trace-id");
  });

  test("getCurrentTraceId 兜底（无 fields.trace_id 时使用全局 trace_id）", () => {
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [new MockTransport()],
      storage: new MemoryStore(),
    });
    const logger = Logger.getInstance()!;
    setCurrentTraceId("fallback-trace-id");

    logger.error("api-error");

    const entry = persistedQueue()[0]!;
    expect(entry.trace_id).toBe("fallback-trace-id");
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
    setCurrentTraceId(generateTraceId());

    logger.error("API_ERROR", {
      method: "POST",
      path: "/api/v1/prediction",
      status: 500,
      duration: 1203.5,
      code: "B0500",
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
    expect(entry.trace_id).toMatch(/^[0-9a-f]{32}$/);
    expect(new Date(entry.timestamp).toISOString()).toBe(entry.timestamp);
  });
});
