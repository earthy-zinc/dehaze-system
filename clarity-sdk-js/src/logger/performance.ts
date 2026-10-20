import type { Logger } from "./Logger";
import type { LogEntry } from "./types";

/**
 * Web 性能采集（对齐前端日志监控改造计划 §3.3.5 / 阶段 2 任务 2.2-2.6）。
 *
 * 覆盖：
 *  - Web Vitals：LCP / INP / CLS（PerformanceObserver）
 *  - 页面加载：FP / FCP / TTFB / dom_ready / load（paint + navigation）
 *  - 长任务：long_task（>50ms 阻塞主线程；>200ms 升 WARN）
 *  - 资源加载失败：resource_error（window 捕获阶段 error）
 *  - SPA 路由切换：route_switch（history pushState/replaceState + popstate）
 *
 * 平台适配：PerformanceObserver / performance 不存在的环境（如部分小程序/RN）降级跳过，
 * 由各端（mobile-dev）在平台层实现等价采集，指标字段名对齐。
 */

/** 长任务是否告警的阈值：超过则升 WARN（50% 采样上报），否则 INFO（不上报） */
const LONG_TASK_WARN_THRESHOLD_MS = 200;

interface PerformanceObserverEntry {
  entryType: string;
  name?: string;
  startTime: number;
  duration?: number;
  value?: number;
  hadRecentInput?: boolean;
}

interface PerformanceObserverLike {
  observe(options?: { type?: string; buffered?: boolean; entryTypes?: string[] }): void;
  disconnect(): void;
}

type ObserverCallback = (
  entries: PerformanceObserverEntry[],
  observer: PerformanceObserverLike
) => void;

interface PerformanceObserverCtor {
  new (callback: ObserverCallback): PerformanceObserverLike;
}

interface NavigationEntry {
  navigationType?: string;
  requestStart?: number;
  responseStart?: number;
  domContentLoadedEventEnd?: number;
  loadEventEnd?: number;
  startTime?: number;
}

/** 从 performance.getEntriesByType 获取首个导航条目（TTFB/dom_ready/load 数据源） */
function navigationEntry(): NavigationEntry | undefined {
  if (typeof performance === "undefined") return undefined;
  const entries = performance.getEntriesByType?.("navigation") as NavigationEntry[] | undefined;
  if (entries && entries.length > 0) return entries[0];
  // 回退到旧版 performance.timing（部分浏览器不支持 navigation timing API）
  const timing = (performance as unknown as { timing?: NavigationEntry }).timing;
  if (!timing) return undefined;
  return timing;
}

/** 性能指标日志的字段负载（navigation_type / resource_url 仅在有值才带上） */
interface MetricFields {
  metric_name: string;
  metric_value: number;
  navigation_type?: string | undefined;
  resource_url?: string | undefined;
}

/** 上报一条性能指标日志（默认 INFO；长任务超阈值由调用方升 WARN） */
function reportMetric(
  logger: Logger,
  level: "INFO" | "WARN",
  message: string,
  fields: MetricFields
): void {
  const entry: Partial<LogEntry> = {
    type: "performance",
    metric_name: fields.metric_name,
    metric_value: fields.metric_value,
  };
  if (fields.navigation_type !== undefined) entry.navigation_type = fields.navigation_type;
  if (fields.resource_url !== undefined) entry.resource_url = fields.resource_url;
  logger.log(level, message, entry);
}

/**
 * 初始化 Web 性能采集，返回清理函数（销毁 Logger 时断开所有 observer 与事件监听）。
 * 环境不支持时直接返回空清理函数，不抛错。
 */
export function initWebPerformanceMonitoring(logger: Logger): () => void {
  const cleanups: Array<() => void> = [];

  const obs = globalThis.PerformanceObserver as unknown as PerformanceObserverCtor | undefined;
  const supportsObserver = typeof obs === "function";

  /** 注册 PerformanceObserver；不支持或注册失败时静默跳过（降级） */
  function observe(entryType: string, callback: ObserverCallback, buffered = true): void {
    if (!supportsObserver) return;
    try {
      const observer = new obs((list, o) => {
        const entries = (Array.isArray(list) ? list : []) as unknown as PerformanceObserverEntry[];
        callback(entries, o);
      });
      observer.observe({ type: entryType, buffered });
      cleanups.push(() => observer.disconnect());
    } catch {
      // 当前环境不支持该 entryType，跳过
    }
  }

  // ---- Web Vitals ----
  observe("largest-contentful-paint", (entries) => {
    const last = entries[entries.length - 1];
    if (last && last.startTime > 0) {
      reportMetric(logger, "INFO", "PERFORMANCE_LCP", {
        metric_name: "lcp",
        metric_value: last.startTime,
        navigation_type: navigationEntry()?.navigationType,
      });
    }
  });

  // INP：event 类型取主线程交互最大耗时（近似 INP，hadRecentInput 过滤滚动期间误报）
  let inpValue = 0;
  observe(
    "event",
    (entries) => {
      for (const entry of entries) {
        if (entry.hadRecentInput) continue;
        const duration = entry.duration ?? 0;
        if (duration > inpValue) inpValue = duration;
      }
    },
    false
  );
  // INP 需持续观测，在 pagehide 时上报最终值（页面离开前输出）
  const onInpHide = () => {
    if (inpValue > 0) {
      reportMetric(logger, "INFO", "PERFORMANCE_INP", {
        metric_name: "inp",
        metric_value: inpValue,
      });
    }
  };
  window.addEventListener("pagehide", onInpHide, { once: true });
  cleanups.push(() => window.removeEventListener("pagehide", onInpHide));

  // CLS：layout-shift 累积（不含最近输入引起的偏移）
  let clsValue = 0;
  observe("layout-shift", (entries) => {
    for (const entry of entries) {
      if (!entry.hadRecentInput) clsValue += entry.value ?? 0;
    }
  });
  const onClsHide = () => {
    if (clsValue > 0) {
      reportMetric(logger, "INFO", "PERFORMANCE_CLS", {
        metric_name: "cls",
        metric_value: clsValue,
      });
    }
  };
  window.addEventListener("pagehide", onClsHide, { once: true });
  cleanups.push(() => window.removeEventListener("pagehide", onClsHide));

  // ---- 页面加载：FP / FCP（paint）----
  observe("paint", (entries) => {
    for (const entry of entries) {
      if (entry.name === "first-paint") {
        reportMetric(logger, "INFO", "PERFORMANCE_FP", {
          metric_name: "fp",
          metric_value: entry.startTime,
          navigation_type: navigationEntry()?.navigationType,
        });
      } else if (entry.name === "first-contentful-paint") {
        reportMetric(logger, "INFO", "PERFORMANCE_FCP", {
          metric_name: "fcp",
          metric_value: entry.startTime,
          navigation_type: navigationEntry()?.navigationType,
        });
      }
    }
  });

  // ---- 页面加载：TTFB / dom_ready / load（navigation）----
  const nav = navigationEntry();
  if (nav && nav.responseStart !== undefined && nav.requestStart !== undefined) {
    reportMetric(logger, "INFO", "PERFORMANCE_TTFB", {
      metric_name: "ttfb",
      metric_value: nav.responseStart - nav.requestStart,
      navigation_type: nav.navigationType,
    });
  }
  if (typeof performance !== "undefined") {
    const navStart = performance.timeOrigin ?? nav?.startTime ?? 0;
    if (nav?.domContentLoadedEventEnd !== undefined) {
      reportMetric(logger, "INFO", "PERFORMANCE_DOM_READY", {
        metric_name: "dom_ready",
        metric_value: nav.domContentLoadedEventEnd - navStart,
        navigation_type: nav.navigationType,
      });
    }
    if (nav?.loadEventEnd !== undefined) {
      reportMetric(logger, "INFO", "PERFORMANCE_LOAD", {
        metric_name: "load",
        metric_value: nav.loadEventEnd - navStart,
        navigation_type: nav.navigationType,
      });
    }
  }

  // ---- 长任务（>50ms 阻塞主线程）----
  observe("longtask", (entries) => {
    for (const entry of entries) {
      const duration = entry.duration ?? 0;
      // >50ms 即视为长任务；超过 200ms 升 WARN（50% 采样），否则 INFO（不上报）
      const level = duration > LONG_TASK_WARN_THRESHOLD_MS ? "WARN" : "INFO";
      reportMetric(logger, level, "PERFORMANCE_LONG_TASK", {
        metric_name: "long_task",
        metric_value: duration,
      });
    }
  });

  // ---- 资源加载失败（resource_error）：window 捕获阶段 error，带 resource_url。
  //      与 Logger.registerGlobalHandlers 的 WARN resource 错误共用同一事件，此处不再重复注册，
  //      由该处理器补充 resource_url 字段（见 Logger.ts）。 ----
  // ---- SPA 路由切换耗时（history pushState/replaceState + popstate）----
  const history = window.history;
  let routeStart = performance?.now?.() ?? 0;

  function startRouteTiming(): void {
    routeStart = performance?.now?.() ?? Date.now();
  }

  function reportRouteSwitch(): void {
    const duration = performance?.now ? performance.now() - routeStart : Date.now() - routeStart;
    reportMetric(logger, "INFO", "PERFORMANCE_ROUTE_SWITCH", {
      metric_name: "route_switch",
      metric_value: duration,
    });
  }

  if (history && typeof history.pushState === "function") {
    const patch = (type: "pushState" | "replaceState") => {
      const original = history[type].bind(history) as (...args: unknown[]) => void;
      (history as unknown as Record<string, unknown>)[type] = (...args: unknown[]) => {
        startRouteTiming();
        original(...args);
        // 路由切换在事件循环下次任务结算，近似采集切换耗时
        setTimeout(reportRouteSwitch, 0);
        return undefined;
      };
      cleanups.push(() => {
        (history as unknown as Record<string, unknown>)[type] = original;
      });
    };
    patch("pushState");
    patch("replaceState");
  }
  const onPopState = () => {
    startRouteTiming();
    setTimeout(reportRouteSwitch, 0);
  };
  window.addEventListener("popstate", onPopState);
  cleanups.push(() => window.removeEventListener("popstate", onPopState));

  return () => {
    for (const cleanup of cleanups) cleanup();
  };
}
