import { afterEach, describe, expect, it, vi } from "vitest";
import AiConversationAPI, { type MessageStreamHandlers } from "@/api/ai-conversation";
import type {
  InterruptData,
  InterruptEvent,
  MessageResumeForm,
  Plan,
  PlanPayload,
  PlanRevision,
  PlanTask,
} from "@/api/ai-conversation/model";

/**
 * SSE 流式事件分发与传输注入单元测试。
 *
 * 通过 `sendMessage`/`reconnectStream` 的 `options.fetchImpl` 注入假 fetch 返回手工拼装的
 * SSE 原文，不再 mock 全局 `fetch`（注入生效后全局 fetch 完全不被触碰）。锁定：
 * 1. 事件 ID 透出（`onEventId`）——断线重连 Last-Event-ID 的精度依赖
 * 2. 各事件类型的 payload 分发——流式消息与断线重连的命脉路径
 * 3. plan 事件（`Plan`）与 plan_approve 中断计划（`PlanPayload`）同形、仅 `phase` 为事件增量
 * 4. InterruptData 全 camelCase 键透传
 */

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

/** 构造 SSE 响应体：把原文按固定字节数切块，模拟网络分片以验证跨 chunk 缓冲拼接 */
function sseResponse(text: string, chunkSize = 7): Response {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return {
    ok: true,
    headers: {
      get: (key: string) => (key.toLowerCase() === "content-type" ? "text/event-stream" : null),
    },
    body: {
      getReader: () => ({
        read: async () => {
          if (offset >= bytes.length) return { done: true, value: undefined };
          const value = bytes.slice(offset, offset + chunkSize);
          offset += value.length;
          return { done: false, value };
        },
      }),
    },
  } as unknown as Response;
}

/** 以 SSE 原文驱动一次 sendMessage（经 fetchImpl 注入假 fetch），返回流结束的 Promise */
function stream(text: string, handlers: MessageStreamHandlers, chunkSize?: number): Promise<void> {
  const fetchImpl = (async () => sseResponse(text, chunkSize)) as typeof fetch;
  return new Promise<void>((resolve, reject) => {
    AiConversationAPI.sendMessage(
      1,
      { content: "你好" },
      {
        ...handlers,
        onClose: () => {
          handlers.onClose?.();
          resolve();
        },
        onNetworkError: (error) => reject(error),
      },
      { fetchImpl }
    );
  });
}

/** 拼装一个 content_block.delta 事件（可选是否携带 id 行） */
function deltaEvent(id: string | null, text: string): string {
  const idLine = id === null ? "" : `id: ${id}\n`;
  const data = JSON.stringify({ index: 0, delta: { type: "text_delta", text } });
  return `${idLine}event: content_block.delta\ndata: ${data}\n\n`;
}

describe("SSE 事件 ID 透出", () => {
  it("按事件顺序透出每个 id", async () => {
    const ids: string[] = [];
    await stream(deltaEvent("1", "A") + deltaEvent("2", "B") + deltaEvent("3", "C"), {
      onEventId: (id) => ids.push(id),
    });
    expect(ids).toEqual(["1", "2", "3"]);
  });

  it("缺 id 帧不触发 onEventId，但事件本身照常分发", async () => {
    const ids: string[] = [];
    const texts: string[] = [];
    await stream(deltaEvent(null, "无 ID"), {
      onEventId: (id) => ids.push(id),
      onContentBlockDelta: (data) => texts.push(data.delta.text ?? ""),
    });
    expect(ids).toEqual([]);
    expect(texts).toEqual(["无 ID"]);
  });

  it("id 回调早于同类事件回调触发", async () => {
    const order: string[] = [];
    await stream(deltaEvent("9", "X"), {
      onEventId: () => order.push("id"),
      onContentBlockDelta: () => order.push("delta"),
    });
    expect(order).toEqual(["id", "delta"]);
  });

  it("未以空行结尾的残缺帧不投递（断流时不交付半截事件）", async () => {
    // SSE 以空行作为事件分派边界，末尾残缺帧（如网络中断造成的半截事件）不应交付，
    // 否则调用方会收到不完整 payload。此处锁定该契约，避免后续"补 flush"改坏。
    const ids: string[] = [];
    const truncated = `id: 2\nevent: content_block.delta\ndata: ${JSON.stringify({
      index: 0,
      delta: { type: "text_delta", text: "尾帧" },
    })}`;
    await stream(deltaEvent("1", "A") + truncated, { onEventId: (id) => ids.push(id) });
    expect(ids).toEqual(["1"]);
  });

  it("payload 非 JSON 时丢弃该事件，但不影响 id 记录", async () => {
    const ids: string[] = [];
    let deltaCount = 0;
    await stream("id: 5\nevent: content_block.delta\ndata: 不是JSON\n\n", {
      onEventId: (id) => ids.push(id),
      onContentBlockDelta: () => (deltaCount += 1),
    });
    expect(ids).toEqual(["5"]);
    expect(deltaCount).toBe(0);
  });
});

describe("SSE 事件分发", () => {
  it("按 event 类型分发到对应回调并解析 payload", async () => {
    const start = vi.fn();
    const blockStart = vi.fn();
    const blockDelta = vi.fn();
    const blockStop = vi.fn();
    const thought = vi.fn();
    const plan = vi.fn();
    const suggestions = vi.fn();
    const interrupt = vi.fn();
    const error = vi.fn();
    const end = vi.fn();

    const raw = [
      `id: 1\nevent: message.start\ndata: ${JSON.stringify({
        messageId: 11,
        conversationId: 1,
        model: "qwen",
      })}\n\n`,
      `id: 2\nevent: content_block.start\ndata: ${JSON.stringify({ index: 0, type: "text" })}\n\n`,
      deltaEvent("3", "你好"),
      `id: 4\nevent: content_block.stop\ndata: ${JSON.stringify({ index: 0 })}\n\n`,
      `id: 5\nevent: thought\ndata: ${JSON.stringify({
        position: 1,
        thought: "分析",
        status: 1,
      })}\n\n`,
      `id: 6\nevent: plan\ndata: ${JSON.stringify({ tasks: [{ id: 1, description: "步骤" }] })}\n\n`,
      `id: 7\nevent: suggestions\ndata: ${JSON.stringify({ questions: [{ question: "然后呢" }] })}\n\n`,
      `id: 8\nevent: interrupt\ndata: ${JSON.stringify({
        type: "confirm",
        data: { recommendation: { algorithm: { id: 2, name: "算法" }, reason: "原因" } },
      })}\n\n`,
      `id: 9\nevent: error\ndata: ${JSON.stringify({ code: "B0001", message: "模型超载" })}\n\n`,
      `id: 10\nevent: message.end\ndata: ${JSON.stringify({
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 2, cachedInputTokens: 0, credits: 3 },
      })}\n\n`,
    ].join("");

    await stream(raw, {
      onStart: start,
      onContentBlockStart: blockStart,
      onContentBlockDelta: blockDelta,
      onContentBlockStop: blockStop,
      onThought: thought,
      onPlan: plan,
      onSuggestions: suggestions,
      onInterrupt: interrupt,
      onError: error,
      onEnd: end,
    });

    expect(start).toHaveBeenCalledWith({ messageId: 11, conversationId: 1, model: "qwen" });
    expect(blockStart).toHaveBeenCalledWith({ index: 0, type: "text" });
    expect(blockDelta).toHaveBeenCalledWith({
      index: 0,
      delta: { type: "text_delta", text: "你好" },
    });
    expect(blockStop).toHaveBeenCalledWith({ index: 0 });
    expect(thought).toHaveBeenCalledWith({ position: 1, thought: "分析", status: 1 });
    expect(plan).toHaveBeenCalledWith({ tasks: [{ id: 1, description: "步骤" }] });
    expect(suggestions).toHaveBeenCalledWith({ questions: [{ question: "然后呢" }] });
    expect(interrupt.mock.calls[0]?.[0]?.type).toBe("confirm");
    expect(error).toHaveBeenCalledWith({ code: "B0001", message: "模型超载" });
    expect(end.mock.calls[0]?.[0]?.stopReason).toBe("end_turn");
  });

  it("未知事件类型与心跳不抛错（ping 仅触发 onPing）", async () => {
    const ping = vi.fn();
    const delta = vi.fn();
    await stream(
      `id: 1\nevent: ping\ndata: {}\n\n` + `id: 2\nevent: 未来新增事件\ndata: {"a":1}\n\n`,
      { onPing: ping, onContentBlockDelta: delta }
    );
    expect(ping).toHaveBeenCalledTimes(1);
    expect(delta).not.toHaveBeenCalled();
  });

  it("非 text/event-stream 响应走直返分支，不触发任何事件回调", async () => {
    const fetchImpl = (async () =>
      ({
        ok: true,
        headers: { get: () => "application/json" },
        text: async () => `{"code":"00000","data":{}}`,
      }) as unknown as Response) as typeof fetch;

    const start = vi.fn();
    const closed = vi.fn();
    await new Promise<void>((resolve) => {
      AiConversationAPI.sendMessage(
        1,
        { content: "你好" },
        {
          onStart: start,
          onClose: () => {
            closed();
            resolve();
          },
        },
        { fetchImpl }
      );
    });
    expect(start).not.toHaveBeenCalled();
    expect(closed).toHaveBeenCalledTimes(1);
  });
});

describe("SSE 传输注入", () => {
  it("注入 fetchImpl 后由注入实现承载请求，全局 fetch 不被触碰", async () => {
    const globalSpy = vi.fn(() => {
      throw new Error("全局 fetch 不应被调用");
    });
    globalThis.fetch = globalSpy as unknown as typeof fetch;

    const injected = vi.fn(async () => sseResponse(""));
    const done = new Promise<void>((resolve) => {
      AiConversationAPI.sendMessage(
        1,
        { content: "注入" },
        { onClose: () => resolve() },
        { fetchImpl: injected }
      );
    });
    await done;

    expect(injected).toHaveBeenCalledTimes(1);
    expect(globalSpy).not.toHaveBeenCalled();
  });
});

describe("plan 事件与 plan_approve 中断的统一契约", () => {
  it("plan 事件任务项透传 camelCase（dependsOn/paradigm/toolHint/result/revisions）", async () => {
    const plan = vi.fn<(data: Plan) => void>();
    const task: PlanTask = {
      id: "t1",
      description: "步骤",
      dependsOn: ["t0"],
      status: "pending",
      paradigm: "react",
      toolHint: "dehaze",
      result: "ok",
    };
    const revision: PlanRevision = {
      revisionNo: 1,
      reason: "子任务失败",
      changedTaskIds: ["t1"],
    };
    const payload: Plan = {
      tasks: [task],
      status: "pending",
      revisions: [revision],
      phase: "plan",
    };
    await stream(`event: plan\ndata: ${JSON.stringify(payload)}\n\n`, { onPlan: plan });
    expect(plan).toHaveBeenCalledWith(payload);
    expect(plan.mock.calls[0]?.[0]?.tasks[0]).toHaveProperty("dependsOn");
    expect(plan.mock.calls[0]?.[0]?.tasks[0]).not.toHaveProperty("depends_on");
  });

  it("plan_approve 中断 data.plan 为 PlanPayload 形状（dependsOn/toolHint，无 phase）", async () => {
    const interrupt = vi.fn<(data: InterruptEvent) => void>();
    const task: PlanTask = {
      id: "t1",
      description: "步骤",
      dependsOn: ["t0"],
      toolHint: "dehaze",
      paradigm: "react",
      result: "ok",
    };
    const plan: PlanPayload = { tasks: [task], status: "pending", revisions: [] };
    await stream(
      `event: interrupt\ndata: ${JSON.stringify({ type: "plan_approve", data: { plan } })}\n\n`,
      { onInterrupt: interrupt }
    );
    const data = interrupt.mock.calls[0]?.[0]?.data;
    expect(data?.plan).toEqual(plan);
    expect(data?.plan?.tasks[0]).toHaveProperty("dependsOn");
    expect(data?.plan?.tasks[0]).not.toHaveProperty("depends_on");
    expect(data?.plan).not.toHaveProperty("phase");
  });

  it("InterruptData 按统一后的 camelCase 键透传（quota/写冲突/授权/异步等待）", async () => {
    const interrupt = vi.fn<(data: InterruptEvent) => void>();
    const quota: InterruptData = {
      upgradeTip: "升级会员",
      usedDaily: 10,
      dailyLimit: 10,
      usedMonthly: 100,
      monthlyLimit: 1000,
      quotaDataError: false,
    };
    const writeConflict: InterruptData = {
      confirmKind: "dangerous_op",
      action: "write_conflict",
      tool: "write_file",
      resource: "report.md",
      previousWriter: "agent-A",
      impact: "覆盖写入",
      reason: "写冲突",
    };
    const permission: InterruptData = {
      confirmKind: "tool_permission",
      tool: "shell_exec",
      reason: "危险操作",
      detail: "需授权",
    };
    const algorithm: InterruptData = {
      confirmKind: "algorithm_recommend",
      artifactId: 9,
      recommendation: {
        recommendationId: 1,
        algorithmId: 9,
        algorithmName: "DCP",
        reason: "适合浓雾",
        effectDescription: "提升清晰度",
      },
      alternatives: [{ algorithmId: 3, algorithmName: "AOD", matchScore: 0.8, reason: "备选" }],
      imageFeatures: { hazeLevel: 2, sceneType: "outdoor", lighting: "bright" },
    };
    const asyncWait: InterruptData = {
      taskId: "task-1",
      taskType: "image_process",
      estDuration: "30s",
      imageCount: 3,
    };

    for (const [type, data] of [
      ["quota", quota],
      ["confirm", writeConflict],
      ["confirm", permission],
      ["confirm", algorithm],
      ["async_wait", asyncWait],
    ] as const) {
      await stream(`event: interrupt\ndata: ${JSON.stringify({ type, data })}\n\n`, {
        onInterrupt: interrupt,
      });
    }

    expect(interrupt.mock.calls.map(([evt]) => evt.data)).toEqual([
      quota,
      writeConflict,
      permission,
      algorithm,
      asyncWait,
    ]);
  });
});

describe("断线重连 Last-Event-ID 请求头", () => {
  /** 驱动一次 reconnectStream（经 fetchImpl 注入假 fetch），返回 fetch mock 与流结束 Promise */
  function reconnect(lastEventId: string) {
    const fetchMock = vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) =>
      sseResponse("")
    );
    const done = new Promise<void>((resolve) => {
      AiConversationAPI.reconnectStream(
        1,
        "stream-1",
        lastEventId,
        { onClose: () => resolve() },
        { fetchImpl: fetchMock as unknown as typeof fetch }
      );
    });
    return { fetchMock, done };
  }

  it("携带上次事件 ID 作为 Last-Event-ID", async () => {
    const { fetchMock, done } = reconnect("7");
    await done;
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(headers["Last-Event-ID"]).toBe("7");
  });

  it("上次事件 ID 为空时不带该请求头（服务端按 0 重放全量）", async () => {
    const { fetchMock, done } = reconnect("");
    await done;
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(headers["Last-Event-ID"]).toBeUndefined();
  });
});

describe("resume 上行请求体键名（对外契约 camelCase）", () => {
  /** 驱动一次 resumeMessage（经 fetchImpl 注入假 fetch），返回 fetch mock 与流结束 Promise */
  function resume(data: MessageResumeForm) {
    const fetchMock = vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) =>
      sseResponse("")
    );
    const done = new Promise<void>((resolve) => {
      AiConversationAPI.resumeMessage(
        7,
        data,
        { onClose: () => resolve() },
        { fetchImpl: fetchMock as unknown as typeof fetch }
      );
    });
    return { fetchMock, done };
  }

  it("planEdit 外层以 plan_edit 上行，add 内层为 camelCase（dependsOn/toolHint/paradigm）", async () => {
    const planEdit: MessageResumeForm["planEdit"] = {
      remove: ["t2"],
      reorder: ["t1"],
      add: { description: "新增任务", dependsOn: ["t0"], toolHint: "dehaze", paradigm: "react" },
    };
    const { fetchMock, done } = resume({ planEdit });
    await done;

    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(String(url).endsWith("/api/v1/ai/messages/7/resume")).toBe(true);
    expect(init?.method).toBe("POST");
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(body).toEqual({ plan_edit: planEdit });
    // 内层不留旧 snake_case 键
    const raw = JSON.stringify(body);
    expect(raw).not.toContain("depends_on");
    expect(raw).not.toContain("tool_hint");
  });

  it("仅 confirm/params 时不带 plan_edit 键", async () => {
    const { fetchMock, done } = resume({ confirm: true, params: { algorithmId: 9 } });
    await done;
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body)) as Record<string, unknown>;
    expect(body).toEqual({ confirm: true, params: { algorithmId: 9 } });
    expect(body).not.toHaveProperty("plan_edit");
  });
});
