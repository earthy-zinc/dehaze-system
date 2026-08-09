/**
 * 前端日志监控 Logger 性能采集测试（阶段 2，任务 2.2-2.6）
 *
 * 覆盖：Web Vitals（LCP/INP/CLS）、页面加载（FP/FCP/TTFB/dom_ready/load）、
 * 长任务（long_task，>200ms 升 WARN）、SPA 路由切换（route_switch pushState/replaceState/popstate）、
 * 环境降级（无 PerformanceObserver 时跳过）。
 *
 * 纯前端逻辑，通过 pnpm test:unit 运行（不依赖后端登录）。
 * 通过 mock window / performance / PerformanceObserver 全局对象驱动采集并断言。
 */
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { Logger } from "@/logger";
import { CaptureTransport, MemoryStore } from "#/unit/logger/helpers";

type PerfEntry = {
  entryType: string;
  name?: string;
  startTime?: number;
  duration?: number;
  value?: number;
  hadRecentInput?: boolean;
};

class MockPerformanceObserver {
  static instances: MockPerformanceObserver[] = [];
  entryType: string | undefined;
  callback: (entries: PerfEntry[], observer: MockPerformanceObserver) => void;

  constructor(callback: (entries: PerfEntry[], observer: MockPerformanceObserver) => void) {
    this.callback = callback;
    MockPerformanceObserver.instances.push(this);
  }

  observe(options: { type?: string; buffered?: boolean }): void {
    this.entryType = options.type;
    const buffered = (globalThis as any).__perfBuffered ?? [];
    const pending = (globalThis as any).__perfPending ?? [];
    const matched = [...buffered, ...pending].filter((e) => e.entryType === this.entryType);
    (globalThis as any).__perfPending = [];
    this.callback(matched, this);
  }

  disconnect(): void {}
}

let listenerMap = new Map<string, Set<EventListener>>();
let historyMethods: {
  pushState: (...a: unknown[]) => void;
  replaceState: (...a: unknown[]) => void;
};

function installWindowMock(): void {
  listenerMap = new Map<string, Set<EventListener>>();
  historyMethods = {
    pushState: () => {},
    replaceState: () => {},
  };
  const mockWindow = {
    location: { pathname: "/prediction/history", search: "" },
    addEventListener: (type: string, handler: EventListener) => {
      if (!listenerMap.has(type)) listenerMap.set(type, new Set());
      listenerMap.get(type)!.add(handler);
    },
    removeEventListener: (type: string, handler: EventListener) => {
      listenerMap.get(type)?.delete(handler);
    },
    history: historyMethods,
  };
  (globalThis as any).window = mockWindow;
  (globalThis as any).addEventListener = mockWindow.addEventListener;
  (globalThis as any).removeEventListener = mockWindow.removeEventListener;
  (globalThis as any).location = mockWindow.location;
}

function cleanupWindowMock(): void {
  delete (globalThis as any).window;
  delete (globalThis as any).addEventListener;
  delete (globalThis as any).removeEventListener;
  delete (globalThis as any).location;
  delete (globalThis as any).PerformanceObserver;
  delete (globalThis as any).performance;
  delete (globalThis as any).__perfPending;
  delete (globalThis as any).__perfBuffered;
}

function installPerformanceMock(): void {
  (globalThis as any).performance = {
    now: () => Date.now(),
    timeOrigin: 100,
    getEntriesByType: (type: string) => {
      if (type === "navigation") {
        return [
          {
            navigationType: "navigate",
            requestStart: 120,
            responseStart: 480,
            domContentLoadedEventEnd: 900,
            loadEventEnd: 1500,
            startTime: 100,
          },
        ];
      }
      return [];
    },
  };
}

function emitPerfEntry(entry: PerfEntry): void {
  const pending = (globalThis as any).__perfPending ?? [];
  pending.push(entry);
  (globalThis as any).__perfPending = pending;
  for (const instance of MockPerformanceObserver.instances) {
    if (instance.entryType === entry.entryType && instance.callback) {
      instance.callback([entry], instance);
    }
  }
}

function installCaptureLogger() {
  const transport = new CaptureTransport();
  Logger.install({
    app: "react",
    appVersion: "1.0.0",
    transports: [transport],
    storage: new MemoryStore(),
  });
  return transport;
}

beforeEach(() => {
  Logger.reset();
  MockPerformanceObserver.instances = [];
  (globalThis as any).__perfBuffered = [];
  (globalThis as any).__perfPending = [];
});

afterEach(() => {
  Logger.reset();
  cleanupWindowMock();
});

describe("性能采集：Web Vitals（LCP/INP/CLS）", () => {
  test("LCP 取多条中最后一条上报", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    // 多条 LCP：浏览器规范仅取最后一条作为最终 LCP
    emitPerfEntry({ entryType: "largest-contentful-paint", startTime: 1500 });
    emitPerfEntry({ entryType: "largest-contentful-paint", startTime: 2500 });
    emitPerfEntry({ entryType: "largest-contentful-paint", startTime: 2200 });

    const lcpLogs = transport.logs.filter((l) => l.metric_name === "lcp");
    // 每条 LCP entry 都会触发一次回调，但代码只上报最后一条（last.startTime > 0）
    // 这里验证最后上报的 metric_value 与最后一条 entry 一致
    const last = lcpLogs[lcpLogs.length - 1];
    expect(last?.type).toBe("performance");
    expect(last?.metric_name).toBe("lcp");
    expect(last?.metric_value).toBe(2200);
    expect(last?.navigation_type).toBe("navigate");
  });

  test("LCP startTime <= 0 时不上报（避免无效指标）", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    emitPerfEntry({ entryType: "largest-contentful-paint", startTime: 0 });

    expect(transport.logs.some((l) => l.metric_name === "lcp")).toBe(false);
  });

  test("CLS 聚合布局偏移值（无量纲），pagehide 触发最终上报", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    emitPerfEntry({ entryType: "layout-shift", value: 0.1 });
    emitPerfEntry({ entryType: "layout-shift", value: 0.2 });
    // 触发 pagehide 上报最终 CLS
    for (const handler of listenerMap.get("pagehide") ?? []) {
      handler({} as Event);
    }

    const cls = transport.logs.find((l) => l.metric_name === "cls");
    expect(cls?.metric_name).toBe("cls");
    expect(cls?.metric_value).toBeCloseTo(0.3);
  });

  test("CLS 跳过 hadRecentInput=true 的偏移（去噪）", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    // 用户输入引起的布局偏移应被过滤
    emitPerfEntry({ entryType: "layout-shift", value: 0.5, hadRecentInput: true });
    emitPerfEntry({ entryType: "layout-shift", value: 0.2, hadRecentInput: false });
    for (const handler of listenerMap.get("pagehide") ?? []) {
      handler({} as Event);
    }

    const cls = transport.logs.find((l) => l.metric_name === "cls");
    expect(cls?.metric_value).toBeCloseTo(0.2);
  });

  test("CLS=0 时不上报（避免无意义日志）", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    // 全部 hadRecentInput 被过滤后 clsValue=0
    emitPerfEntry({ entryType: "layout-shift", value: 0.5, hadRecentInput: true });
    for (const handler of listenerMap.get("pagehide") ?? []) {
      handler({} as Event);
    }

    expect(transport.logs.some((l) => l.metric_name === "cls")).toBe(false);
  });
});

describe("性能采集：页面加载（FP/FCP/TTFB/dom_ready/load）", () => {
  test("paint 事件上报 FP 与 FCP", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    emitPerfEntry({ entryType: "paint", name: "first-paint", startTime: 300 });
    emitPerfEntry({ entryType: "paint", name: "first-contentful-paint", startTime: 500 });

    const fp = transport.logs.find((l) => l.metric_name === "fp");
    const fcp = transport.logs.find((l) => l.metric_name === "fcp");
    expect(fp?.metric_value).toBe(300);
    expect(fp?.navigation_type).toBe("navigate");
    expect(fcp?.metric_value).toBe(500);
    expect(fcp?.navigation_type).toBe("navigate");
  });

  test("navigation timing 上报 TTFB/dom_ready/load", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    const ttfb = transport.logs.find((l) => l.metric_name === "ttfb");
    const domReady = transport.logs.find((l) => l.metric_name === "dom_ready");
    const load = transport.logs.find((l) => l.metric_name === "load");
    // responseStart(480) - requestStart(120)
    expect(ttfb?.metric_value).toBe(480 - 120);
    expect(ttfb?.navigation_type).toBe("navigate");
    // domContentLoadedEventEnd(900) - timeOrigin(100)
    expect(domReady?.metric_value).toBe(900 - 100);
    // loadEventEnd(1500) - timeOrigin(100)
    expect(load?.metric_value).toBe(1500 - 100);
  });
});

describe("性能采集：长任务（long_task）", () => {
  test(">200ms 长任务升 WARN，≤200ms 保持 INFO", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    emitPerfEntry({ entryType: "longtask", duration: 120 });
    emitPerfEntry({ entryType: "longtask", duration: 260 });

    const normal = transport.logs.find(
      (l) => l.metric_name === "long_task" && l.metric_value === 120
    );
    const warn = transport.logs.find(
      (l) => l.metric_name === "long_task" && l.metric_value === 260
    );
    expect(normal?.level).toBe("INFO");
    expect(warn?.level).toBe("WARN");
  });

  test("恰好 200ms 长任务保持 INFO（边界值）", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    emitPerfEntry({ entryType: "longtask", duration: 200 });

    const boundary = transport.logs.find(
      (l) => l.metric_name === "long_task" && l.metric_value === 200
    );
    expect(boundary?.level).toBe("INFO");
  });
});

describe("性能采集：SPA 路由切换（route_switch）", () => {
  test("history.pushState 触发 route_switch 上报", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    const pushState = (globalThis as any).window.history.pushState;
    pushState({}, "", "/next");

    return new Promise<void>((resolve) => {
      setTimeout(() => {
        const route = transport.logs.filter((l) => l.metric_name === "route_switch");
        expect(route.length).toBeGreaterThanOrEqual(1);
        const last = route[route.length - 1];
        expect(last?.type).toBe("performance");
        expect(last?.metric_name).toBe("route_switch");
        expect(typeof last?.metric_value).toBe("number");
        // 切换耗时为非负数且合理（不应该是 NaN 或负数）
        expect(last!.metric_value!).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(last!.metric_value)).toBe(true);
        resolve();
      }, 10);
    });
  });

  test("history.replaceState 同样触发 route_switch 上报", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    const replaceState = (globalThis as any).window.history.replaceState;
    replaceState({}, "", "/replaced");

    return new Promise<void>((resolve) => {
      setTimeout(() => {
        const route = transport.logs.find((l) => l.metric_name === "route_switch");
        expect(route?.type).toBe("performance");
        expect(route?.metric_name).toBe("route_switch");
        expect(Number.isFinite(route?.metric_value)).toBe(true);
        resolve();
      }, 10);
    });
  });

  test("popstate 事件触发 route_switch 上报", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    // 模拟浏览器后退触发 popstate
    for (const handler of listenerMap.get("popstate") ?? []) {
      handler({} as Event);
    }

    return new Promise<void>((resolve) => {
      setTimeout(() => {
        const route = transport.logs.find((l) => l.metric_name === "route_switch");
        expect(route?.type).toBe("performance");
        expect(route?.metric_name).toBe("route_switch");
        expect(Number.isFinite(route?.metric_value)).toBe(true);
        resolve();
      }, 10);
    });
  });

  test("Logger.reset 还原 history 方法，后续 pushState 不再触发 route_switch", () => {
    installWindowMock();
    installPerformanceMock();
    (globalThis as any).PerformanceObserver = MockPerformanceObserver;

    const transport = installCaptureLogger();

    // reset 后 history 应被还原（不再 patch 触发 route_switch）
    Logger.reset();

    // 重新 install 一个新的 capture logger 接管后续日志
    const transport2 = new CaptureTransport();
    Logger.install({
      app: "react",
      appVersion: "1.0.0",
      transports: [transport2],
      storage: new MemoryStore(),
    });

    const pushState = (globalThis as any).window.history.pushState;
    pushState({}, "", "/after-reset");

    return new Promise<void>((resolve) => {
      setTimeout(() => {
        // reset 后的 pushState 不应被 patch 触发 route_switch
        // transport 是 reset 前的，应仍为空（无 route_switch）
        expect(transport.logs.some((l) => l.metric_name === "route_switch")).toBe(false);
        // transport2 是 reset 后新 install 的，会重新 patch pushState 并触发
        expect(transport2.logs.some((l) => l.metric_name === "route_switch")).toBe(true);
        resolve();
      }, 10);
    });
  });
});

describe("性能采集：环境降级", () => {
  test("无 PerformanceObserver 时降级：仅上报无需 observer 的页面加载指标，observer 类指标跳过", () => {
    installWindowMock();
    installPerformanceMock();
    // 不设置 PerformanceObserver，模拟不支持 PerformanceObserver 的环境（如部分小程序/RN）

    const transport = installCaptureLogger();

    // 不抛错；无需 observer 的 TTFB/dom_ready/load 照常上报
    const names = transport.logs.map((l) => l.metric_name).sort();
    expect(names).toEqual(["dom_ready", "load", "ttfb"]);
    // observer 类指标（lcp/fp/fcp/long_task/cls/inp）不采集
    for (const name of ["lcp", "fp", "fcp", "long_task", "cls", "inp"]) {
      expect(transport.logs.some((l) => l.metric_name === name)).toBe(false);
    }
  });
});
