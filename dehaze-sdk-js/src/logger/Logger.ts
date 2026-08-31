import { isBrowser, isRN } from "./env";
import { initWebPerformanceMonitoring } from "./performance";
import { ConsoleTransport } from "./transports";
import type { InstallConfig, LogEntry, LogLevel, LogTransport, LoggerStorage } from "./types";

const QUEUE_KEY = "dehaze_logs";
const MAX_QUEUE = 100;
const INITIAL_BACKOFF_MS = 1_000;
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 20;
const MAX_MESSAGE_LENGTH = 2000;
const MAX_STACK_LENGTH = 8000;
const DEDUP_WINDOW_MS = 10_000;

const REPORT_SAMPLE_RATE: Record<LogLevel, number> = {
  ERROR: 100,
  WARN: 50,
  INFO: 0,
};

function currentUrl(): string {
  // 小程序：wx/uni 同步 API
  const wx = (globalThis as any).wx;
  if (wx?.getCurrentPages) {
    try {
      const pages = wx.getCurrentPages();
      const route = pages?.[pages.length - 1]?.route ?? "";
      return route ? `/${route}` : "";
    } catch {
      /* 降级 */
    }
  }
  const uni = (globalThis as any).uni;
  if (uni?.getCurrentPages) {
    try {
      const pages = uni.getCurrentPages();
      const route = pages?.[pages.length - 1]?.route ?? "";
      return route ? `/${route}` : "";
    } catch {
      /* 降级 */
    }
  }
  return isBrowser() && window.location ? window.location.pathname + window.location.search : "";
}

function currentUserAgent(): string {
  const wx = (globalThis as any).wx;
  if (wx?.getSystemInfoSync) {
    try {
      return `wx ${wx.getSystemInfoSync().model ?? ""}`;
    } catch {
      /* 降级 */
    }
  }
  const uni = (globalThis as any).uni;
  if (uni?.getSystemInfoSync) {
    try {
      return `uni ${uni.getSystemInfoSync().model ?? ""}`;
    } catch {
      /* 降级 */
    }
  }
  // RN 的 navigator 仅有 product，无 userAgent
  if (isRN()) return "ReactNative";
  return typeof navigator !== "undefined" ? navigator.userAgent : "";
}

/**
 * 前端日志 SDK 核心：多 transport 架构、全局错误捕获、批量上报、localStorage 离线缓存、采样限流。
 * SDK 不感知环境：应用端按需传 transports（开发只 Console，生产 Console+Remote）。
 */
export class Logger {
  private static instance: Logger | undefined;

  private readonly app: string;
  private readonly appVersion: string | undefined;
  private readonly transports: LogTransport[];
  private readonly storage: LoggerStorage;
  private readonly rateLimitMax: number;
  private readonly rateLimitWindowMs: number;

  private queue: LogEntry[] = [];
  private backoffMs = INITIAL_BACKOFF_MS;
  private flushTimer: ReturnType<typeof setInterval> | undefined;
  private backoffTimer: ReturnType<typeof setTimeout> | undefined;
  private flushing = false;
  private readonly sentTimestamps: number[] = [];
  private readonly registeredHandlers: Array<[EventTarget, string, EventListener]> = [];
  private disposePerformance: (() => void) | undefined;

  // ERROR 去重：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条，
  // 窗口结束时若存在重复则补发一条汇总（dedup_count 标记总次数），避免日志风暴同时保留次数信息
  private lastErrorFingerprint = 0;
  private lastErrorTime = 0;
  private errorDedupCount = 0;
  private lastDedupEntry: LogEntry | undefined;
  private dedupSummaryTimer: ReturnType<typeof setTimeout> | undefined;

  private constructor(config: InstallConfig) {
    this.app = config.app;
    this.appVersion = config.appVersion;
    this.transports = config.transports ?? [new ConsoleTransport()];
    this.storage = config.storage ?? defaultStorage();
    this.rateLimitMax = config.rateLimitMax ?? RATE_LIMIT_MAX;
    this.rateLimitWindowMs = config.rateLimitWindowMs ?? RATE_LIMIT_WINDOW_MS;
    this.loadQueue();
  }

  static install(config: InstallConfig): Logger {
    if (config.react !== undefined) {
      bindReact(config.react);
    }
    if (Logger.instance) {
      Logger.instance.configure(config);
      return Logger.instance;
    }
    const logger = new Logger(config);
    Logger.instance = logger;
    logger.startGlobalHandlers();
    return logger;
  }

  static getInstance(): Logger | undefined {
    return Logger.instance;
  }

  /**
   * 销毁当前实例：清除定时器与全局监听，释放单例。
   * 用于测试隔离、HMR 热更新或应用退出。
   */
  static reset(): void {
    const logger = Logger.instance;
    if (!logger) return;
    logger.disposeGlobalHandlers();
    Logger.instance = undefined;
  }

  /**
   * 二次 install 时更新配置：若传入 transports，先 dispose 旧的全局处理器与定时器
   * （避免日志输出到已失效的 transport），更新 transports 后重新注册，保证全局错误捕获
   * 走新 transport。保留 queue 与限流计数，不丢失未上报日志。
   */
  private configure(config: InstallConfig): void {
    if (config.transports) {
      this.disposeGlobalHandlers();
      this.transports.splice(0, this.transports.length, ...config.transports);
      this.startGlobalHandlers();
    }
  }

  /** 清理全局错误处理器、性能监控、定时器（保留 queue 与限流计数） */
  private disposeGlobalHandlers(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = undefined;
    }
    if (this.backoffTimer) {
      clearTimeout(this.backoffTimer);
      this.backoffTimer = undefined;
    }
    if (this.dedupSummaryTimer) {
      clearTimeout(this.dedupSummaryTimer);
      this.dedupSummaryTimer = undefined;
    }
    this.disposePerformance?.();
    this.disposePerformance = undefined;
    for (const [target, type, handler] of this.registeredHandlers) {
      target.removeEventListener(type, handler);
    }
    this.registeredHandlers.length = 0;
  }

  /** 注册全局错误处理器、性能监控、flush 定时器 */
  private startGlobalHandlers(): void {
    this.registerGlobalHandlers();
    this.startPerformanceMonitoring();
    this.startFlushTimer();
  }

  log(level: LogLevel, message: string, fields: Partial<LogEntry> = {}): void {
    // trace_id 仅由调用方通过 fields.trace_id 显式传入（请求链路从 config.metadata 读取）。
    // 不再读取全局变量：并发请求时全局变量会被覆盖，导致非请求日志 trace_id 串号；
    // 全局错误/性能日志语义上不属于请求链路，trace_id 留空。
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message: truncate(message, MAX_MESSAGE_LENGTH),
      service: "client",
      app: this.app,
      url: fields.url ?? currentUrl(),
      user_agent: fields.user_agent ?? currentUserAgent(),
      trace_id: fields.trace_id ?? "",
      ...fields,
    };
    if (this.appVersion) {
      entry.app_version = this.appVersion;
    }
    if (entry.error_stack) {
      entry.error_stack = truncate(entry.error_stack, MAX_STACK_LENGTH);
    }

    // ERROR 去重：相同 fingerprint 在 10s 窗口内只输出首条，窗口结束时补发汇总
    if (level === "ERROR" && this.shouldDedupError(entry)) {
      return;
    }

    this.emit(entry);
  }

  /** 实际输出日志条目：transport 输出 + 采样 + 限流 + 入队。去重汇总补发也走此路径 */
  private emit(entry: LogEntry): void {
    // 逐条本地输出（Console），不受采样/限流影响
    for (const transport of this.transports) {
      transport.log?.(entry);
    }

    // 采样过滤：仅 ERROR 全量 / WARN 50% / INFO 不上报
    if (Math.random() * 100 > REPORT_SAMPLE_RATE[entry.level]) {
      return;
    }
    // 单设备限流：60s 内最多上报 20 条
    if (!this.allowReport()) {
      return;
    }

    this.enqueue(entry);
  }

  error(message: string, fields: Partial<LogEntry> = {}): void {
    this.log("ERROR", message, fields);
  }

  warn(message: string, fields: Partial<LogEntry> = {}): void {
    this.log("WARN", message, fields);
  }

  info(message: string, fields: Partial<LogEntry> = {}): void {
    this.log("INFO", message, fields);
  }

  /**
   * 在请求入口复用/生成 trace_id，并写入全局变量供日志串联。
   *
   * ⚠️ 仅供宿主在非 SDK 请求链路（如 WebSocket、fetch 直调）中手动管理 trace_id 使用。
   * SDK 内部 Axios 请求链路不再调用此方法（trace_id 绑定到 config.metadata，避免并发覆盖）。
   * 全局变量在并发请求下会被覆盖，不应依赖其作为请求间共享的 trace_id 载体。
   */
  ensureTraceId(): string {
    const traceId = getCurrentTraceId() || generateTraceId();
    setCurrentTraceId(traceId);
    return traceId;
  }

  /**
   * 响应头 X-Trace-Id 回写对齐到全局变量。
   *
   * ⚠️ 仅供宿主手动管理 trace_id 使用。SDK Axios 链路改为对齐到 config.metadata.traceId。
   */
  alignTraceId(traceId: string | undefined): void {
    if (traceId) {
      setCurrentTraceId(traceId);
    }
  }

  /** 暴露当前全局 trace_id，供宿主在非 SDK 链路中读取 */
  getTraceId(): string {
    return getCurrentTraceId();
  }

  private allowReport(): boolean {
    const now = Date.now();
    while (
      this.sentTimestamps.length > 0 &&
      this.sentTimestamps[0]! <= now - this.rateLimitWindowMs
    ) {
      this.sentTimestamps.shift();
    }
    if (this.sentTimestamps.length >= this.rateLimitMax) {
      return false;
    }
    this.sentTimestamps.push(now);
    return true;
  }

  /**
   * ERROR 去重判定：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条。
   * 窗口内重复命中累加计数并跳过输出；新 fingerprint 或窗口过期时补发上一轮汇总。
   * 返回 true 表示该条应被去重跳过，false 表示正常输出。
   */
  private shouldDedupError(entry: LogEntry): boolean {
    const fingerprint = fingerprintHash(entry.message, entry.error_stack);
    const now = Date.now();
    const inWindow = this.lastErrorTime > 0 && now - this.lastErrorTime < DEDUP_WINDOW_MS;

    if (fingerprint === this.lastErrorFingerprint && inWindow) {
      this.errorDedupCount++;
      return true;
    }

    // 新 burst：先补发上一轮汇总（若有重复）
    this.flushDedupSummary();
    this.lastErrorFingerprint = fingerprint;
    this.lastErrorTime = now;
    this.errorDedupCount = 1;
    this.lastDedupEntry = entry;
    this.scheduleDedupSummary();
    return false;
  }

  /** 窗口结束时补发汇总条目：携带 dedup_count 标记本轮总次数，message 标注重复次数 */
  private flushDedupSummary(): void {
    if (this.dedupSummaryTimer) {
      clearTimeout(this.dedupSummaryTimer);
      this.dedupSummaryTimer = undefined;
    }
    const count = this.errorDedupCount;
    const original = this.lastDedupEntry;
    this.errorDedupCount = 0;
    this.lastErrorFingerprint = 0;
    this.lastErrorTime = 0;
    this.lastDedupEntry = undefined;

    // 单次命中无重复时不补发，避免噪声
    if (count <= 1 || !original) {
      return;
    }

    this.emit({
      timestamp: new Date().toISOString(),
      level: "ERROR",
      message: truncate(`${original.message} (10s 内重复 ${count - 1} 次)`, MAX_MESSAGE_LENGTH),
      service: "client",
      app: this.app,
      url: currentUrl(),
      user_agent: currentUserAgent(),
      // 汇总条目沿用首条的 trace_id（首条已由请求链路显式传入），不读全局变量
      trace_id: original.trace_id,
      ...(original.error_type ? { error_type: original.error_type } : {}),
      ...(original.error_source ? { error_source: original.error_source } : {}),
      ...(original.error_stack ? { error_stack: original.error_stack } : {}),
      dedup_count: count,
      ...(this.appVersion ? { app_version: this.appVersion } : {}),
    });
  }

  private scheduleDedupSummary(): void {
    if (this.dedupSummaryTimer) {
      clearTimeout(this.dedupSummaryTimer);
    }
    this.dedupSummaryTimer = setTimeout(() => {
      this.dedupSummaryTimer = undefined;
      this.flushDedupSummary();
    }, DEDUP_WINDOW_MS);
  }

  private enqueue(entry: LogEntry): void {
    if (this.queue.length >= MAX_QUEUE) {
      this.queue.shift();
    }
    this.queue.push(entry);
    this.persistQueue();
    if (this.queue.length >= 10) {
      this.flush();
    }
  }

  private loadQueue(): void {
    try {
      const raw = this.storage.getItem(QUEUE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as LogEntry[];
      if (Array.isArray(parsed)) {
        this.queue = parsed.slice(-MAX_QUEUE);
      }
    } catch {
      // 缓存损坏则丢弃，不阻塞上报
    }
  }

  private persistQueue(): void {
    try {
      this.storage.setItem(QUEUE_KEY, JSON.stringify(this.queue));
    } catch {
      // 存储配额不足时忽略，队列仅保留内存
    }
  }

  async flush(): Promise<void> {
    if (this.flushing || this.queue.length === 0) return;
    const transport = this.transports.find(
      (item): item is LogTransport & { send: NonNullable<LogTransport["send"]> } =>
        typeof item.send === "function"
    );
    if (!transport) return;
    this.flushing = true;
    try {
      const batch = this.queue.splice(0);
      try {
        await transport.send(batch);
        this.persistQueue();
        this.backoffMs = INITIAL_BACKOFF_MS;
      } catch {
        // 上报失败：恢复队列，走指数退避重试
        this.queue.unshift(...batch);
        this.persistQueue();
        this.scheduleBackoff();
      }
    } finally {
      this.flushing = false;
    }
  }

  /** 网络恢复（online 事件）触发补报离线缓存 */
  flushOnOnline(): void {
    this.flush();
  }

  private scheduleBackoff(): void {
    if (this.backoffTimer) return;
    const delay = this.backoffMs;
    this.backoffMs = Math.min(this.backoffMs * 2, 60_000);
    this.backoffTimer = setTimeout(() => {
      this.backoffTimer = undefined;
      this.flush();
    }, delay);
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flush();
    }, 30_000);
  }

  private registerGlobalHandlers(): void {
    // 1. 小程序 wx.onError / uni.onError（同步字符串错误）
    const wx = (globalThis as any).wx;
    const uni = (globalThis as any).uni;
    const onError = wx?.onError ?? uni?.onError;
    if (typeof onError === "function") {
      onError((error: string) => {
        this.error(`小程序脚本异常: ${error}`, {
          error_type: "js",
          error_source: "wx_on_error",
          error_stack: error,
        });
      });
    }
    // 2. React Native ErrorUtils.setGlobalHandler
    // 必须链式调用原 handler：否则 RN dev 红屏与默认崩溃上报被吞，错误无感知
    const ErrorUtils = (globalThis as any).ErrorUtils;
    if (ErrorUtils?.setGlobalHandler) {
      const originalHandler = ErrorUtils.getGlobalHandler?.();
      ErrorUtils.setGlobalHandler((error: Error | undefined, isFatal: boolean) => {
        this.error(
          `RN 未捕获异常${isFatal ? "（致命）" : ""}: ${error?.message ?? String(error)}`,
          {
            error_type: "js",
            error_source: "error_util_global_handler",
            error_stack: error?.stack ?? String(error),
          }
        );
        originalHandler?.(error, isFatal);
      });
    }
    // 3. Web window.onerror / unhandledrejection / online / 资源 error
    if (!isBrowser()) return;

    const onWindowError = (event: ErrorEvent) => {
      if (event.target && event.target !== window) {
        const target = event.target as HTMLElement;
        const resourceUrl =
          (target as HTMLImageElement).src ||
          (target as HTMLScriptElement).src ||
          (target as HTMLLinkElement).href ||
          target.tagName.toLowerCase();
        this.warn(`资源加载失败: ${resourceUrl}`, {
          error_type: "resource",
          error_source: "resource_error",
          resource_url: resourceUrl,
        });
        return;
      }
      // 过滤浏览器良性警告（不构成功能问题，避免误报刷屏）：
      // ResizeObserver loop 是浏览器已知 benign 错误（布局测量循环达上限），规范允许并建议忽略，
      // 常见于 Element Plus 表格/弹层等组件的尺寸自适应场景，无法从应用层根治。
      const message = event.message || "";
      if (
        message.includes("ResizeObserver loop") ||
        message.includes("ResizeObserver loop completed with undelivered notifications")
      ) {
        return;
      }
      this.error(event.message || "Uncaught error", {
        error_type: "js",
        error_source: "window.onerror",
        error_stack: event.error?.stack ?? `${event.message} at ${event.filename}:${event.lineno}`,
      });
    };

    const onRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason;
      this.error(`未处理的 Promise 拒绝: ${reason?.message ?? String(reason)}`, {
        error_type: "promise",
        error_source: "unhandledrejection",
        error_stack: reason?.stack ?? String(reason),
      });
    };

    const onOnline = () => {
      this.flushOnOnline();
    };

    this.addHandler(window, "error", onWindowError as EventListener);
    this.addHandler(window, "unhandledrejection", onRejection as EventListener);
    this.addHandler(window, "online", onOnline as EventListener);
  }

  private addHandler(target: EventTarget, type: string, handler: EventListener): void {
    target.addEventListener(type, handler);
    this.registeredHandlers.push([target, type, handler]);
  }

  /** 启动性能采集。按运行时环境适配：Web Vitals / 小程序 wx.getPerformance / RN process_time。不支持时降级跳过。 */
  private startPerformanceMonitoring(): void {
    // 1. 小程序 wx.getPerformance / uni.getPerformance
    const wx = (globalThis as any).wx;
    const uni = (globalThis as any).uni;
    const perf = wx?.getPerformance?.() ?? uni?.getPerformance?.();
    if (perf) {
      try {
        const entries = perf.getEntries?.() ?? [];
        for (const e of entries) {
          if (e.entryType === "navigation" && e.duration > 0) {
            this.info("小程序页面加载", {
              metric_name: "load",
              metric_value: e.duration,
              type: "performance",
            } as any);
          }
        }
      } catch {
        /* 降级 */
      }
      return;
    }
    // 2. RN Hermes performance.now（进程运行时长）
    if (isRN()) {
      if (globalThis.performance?.now) {
        try {
          this.info("PERFORMANCE_PROCESS_TIME", {
            metric_name: "process_time",
            metric_value: globalThis.performance.now(),
            type: "performance",
          } as any);
        } catch {
          /* 降级 */
        }
      }
      return;
    }
    // 3. Web PerformanceObserver（Web Vitals/页面加载/长任务/路由切换）
    if (!isBrowser()) return;
    this.disposePerformance = initWebPerformanceMonitoring(this);
  }
}

function truncate(value: string, max: number): string {
  return value.length > max ? value.slice(0, max) : value;
}

/** ERROR 去重 fingerprint：message + error_stack 的轻量 hash（djb2 变体，无需强 hash） */
function fingerprintHash(message: string, errorStack: string | undefined): number {
  const str = `${message}|${errorStack ?? ""}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (Math.imul(hash, 31) + str.charCodeAt(i)) | 0;
  }
  return hash;
}

// ---------- trace_id 生成与当前请求 trace 管理 ----------
// 与后端 §4.3 约定一致：uuid hex 32 位无连字符，透传请求头 X-Trace-Id，响应头回写对齐。
let currentTraceId = "";

function randomBytes(count: number): Uint8Array {
  const bytes = new Uint8Array(count);
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.getRandomValues === "function") {
    cryptoObj.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return bytes;
}

export function generateTraceId(): string {
  return Array.from(randomBytes(16), (b) => b.toString(16).padStart(2, "0")).join("");
}

export function getCurrentTraceId(): string {
  return currentTraceId;
}

export function setCurrentTraceId(traceId: string): void {
  currentTraceId = traceId;
}

// ---------- 默认存储 ----------
/** 内存存储兜底：环境无 localStorage（如 SSR、Node 测试、小程序阶段接入前）时使用 */
class MemoryStorage implements LoggerStorage {
  private store = new Map<string, string>();

  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, value);
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }
}

export function defaultStorage(): LoggerStorage {
  // 1. 小程序 wx 同步存储
  const wx = (globalThis as any).wx;
  if (wx?.getStorageSync) {
    return {
      getItem: (k) => {
        try {
          return wx.getStorageSync(k) || null;
        } catch {
          return null;
        }
      },
      setItem: (k, v) => {
        try {
          wx.setStorageSync(k, v);
        } catch {
          /* 配额不足忽略 */
        }
      },
      removeItem: (k) => {
        try {
          wx.removeStorageSync(k);
        } catch {
          /* 忽略 */
        }
      },
    };
  }
  // 2. uni-app 同步存储
  const uni = (globalThis as any).uni;
  if (uni?.getStorageSync) {
    return {
      getItem: (k) => {
        try {
          return uni.getStorageSync(k) || null;
        } catch {
          return null;
        }
      },
      setItem: (k, v) => {
        try {
          uni.setStorageSync(k, v);
        } catch {
          /* 忽略 */
        }
      },
      removeItem: (k) => {
        try {
          uni.removeStorageSync(k);
        } catch {
          /* 忽略 */
        }
      },
    };
  }
  // 3. React Native AsyncStorage（异步，运行时动态 require，不声明依赖）
  try {
    const AsyncStorage = require("@react-native-async-storage/async-storage");
    if (AsyncStorage?.getItem) {
      const cache = new Map<string, string>();
      // 异步水合上次会话遗留的离线队列
      AsyncStorage.getItem("dehaze_logs")
        .then((raw: string | null) => {
          if (raw) cache.set("dehaze_logs", raw);
        })
        .catch(() => {
          /* 水合失败忽略 */
        });
      return {
        getItem: (k) => cache.get(k) ?? null,
        setItem: (k, v) => {
          cache.set(k, v);
          AsyncStorage.setItem(k, v).catch(() => {
            /* 写入失败忽略 */
          });
        },
        removeItem: (k) => {
          cache.delete(k);
          AsyncStorage.removeItem(k).catch(() => {
            /* 忽略 */
          });
        },
      };
    }
  } catch {
    /* 非 RN 环境，require 失败属预期 */
  }
  // 4. Web localStorage（隐私模式被拒绝时兜底内存）
  try {
    const ls = globalThis.localStorage;
    if (ls && typeof ls.getItem === "function") {
      return ls as unknown as LoggerStorage;
    }
  } catch {
    /* localStorage 访问被拒绝 */
  }
  return new MemoryStorage();
}

// ---------- React 实例注入 ----------
// 宿主注入 React 实例（`Logger.install({ react })` 时设置）。ErrorBoundary 依赖 React，
// 但 SDK 自身不声明 react 依赖，保持零依赖。
let reactInstance: unknown;

export function bindReact(react: unknown): void {
  reactInstance = react;
}

export function getReact(): unknown {
  return reactInstance;
}
