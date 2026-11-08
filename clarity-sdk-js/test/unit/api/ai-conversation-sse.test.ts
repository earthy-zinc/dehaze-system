import { afterEach, describe, expect, it, vi } from "vitest";
import AiConversationAPI, { type MessageStreamHandlers } from "@/api/ai-conversation";
import { SSEHttpError } from "@/utils/sse";
import type {
  Block,
  BlockDeltaEvent,
  BlockStartEvent,
  BlockStopEvent,
  Interrupt,
  Message,
  MessageEndEvent,
  MessageResume,
  MessageStartEvent,
  Plan,
  PlanUpdateEvent,
  Usage,
} from "@/api/ai-conversation/model";

/**
 * SSE 流式事件分发与传输注入单元测试（新契约）。
 *
 * 通过 `sendMessage`/`reconnectStream` 的 `options.fetchImpl` 注入假 fetch 返回手工拼装的
 * SSE 原文，不 mock 全局 `fetch`。锁定：
 * 1. 事件 ID 透出（`onEventId`）——断线重连 Last-Event-ID 的精度依赖
 * 2. 新事件名集合（message.start / block.* / plan.update / interrupt / usage / suggestions /
 *    message.end / error / ping）逐一分发，旧事件名 content_block.* / thought 一律丢弃
 * 3. 三种 block（text/thinking/tool_use）的 start/delta/stop 原样透传
 * 4. 终态唯一：message.end 只交付一次，其后的帧不再分发
 */

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

const usage: Usage = { inputTokens: 1, outputTokens: 2, cachedInputTokens: 0, credits: 3 };

/** 构造一条完整 Message（只覆盖用例关心的字段） */
function message(overrides: Partial<Message> = {}): Message {
  return {
    id: 11,
    conversationId: 1,
    role: "assistant",
    status: "running",
    blocks: [],
    usage,
    artifactIds: [],
    memoryIds: [],
    suggestions: [],
    edited: false,
    createTime: "2026-10-21T10:00:00Z",
    ...overrides,
  };
}

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

/** 拼装一帧 SSE 原文（id 传 null 表示无 id 行） */
function frame(id: string | null, event: string, data: unknown): string {
  const idLine = id === null ? "" : `id: ${id}\n`;
  return `${idLine}event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
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

/** 一段 block.delta 帧 */
function deltaFrame(id: string | null, blockId: string, delta: string): string {
  const data: BlockDeltaEvent = { messageId: 11, blockId, delta };
  return frame(id, "block.delta", data);
}

describe("SSE 事件 ID 透出", () => {
  it("按事件顺序透出每个 id", async () => {
    const ids: string[] = [];
    await stream(
      deltaFrame("1", "b1", "A") + deltaFrame("2", "b1", "B") + deltaFrame("3", "b1", "C"),
      { onEventId: (id) => ids.push(id) }
    );
    expect(ids).toEqual(["1", "2", "3"]);
  });

  it("缺 id 帧不触发 onEventId，但事件本身照常分发", async () => {
    const ids: string[] = [];
    const deltas: BlockDeltaEvent[] = [];
    await stream(deltaFrame(null, "b1", "无 ID"), {
      onEventId: (id) => ids.push(id),
      onBlockDelta: (data) => deltas.push(data),
    });
    expect(ids).toEqual([]);
    expect(deltas).toEqual([{ messageId: 11, blockId: "b1", delta: "无 ID" }]);
  });

  it("id 回调早于同类事件回调触发", async () => {
    const order: string[] = [];
    await stream(deltaFrame("9", "b1", "X"), {
      onEventId: () => order.push("id"),
      onBlockDelta: () => order.push("delta"),
    });
    expect(order).toEqual(["id", "delta"]);
  });

  it("未以空行结尾的残缺帧不投递（断流时不交付半截事件）", async () => {
    const ids: string[] = [];
    const truncated = `id: 2\nevent: block.delta\ndata: ${JSON.stringify({
      messageId: 11,
      blockId: "b1",
      delta: "尾帧",
    })}`;
    await stream(deltaFrame("1", "b1", "A") + truncated, { onEventId: (id) => ids.push(id) });
    expect(ids).toEqual(["1"]);
  });

  it("payload 非 JSON 时丢弃该事件，但不影响 id 记录", async () => {
    const ids: string[] = [];
    let deltaCount = 0;
    await stream("id: 5\nevent: block.delta\ndata: 不是JSON\n\n", {
      onEventId: (id) => ids.push(id),
      onBlockDelta: () => (deltaCount += 1),
    });
    expect(ids).toEqual(["5"]);
    expect(deltaCount).toBe(0);
  });
});

describe("SSE 事件分发（新事件名）", () => {
  const startPayload: MessageStartEvent = {
    streamSessionId: "stream-1",
    userMessage: message({ id: 10, role: "user", status: "completed" }),
    assistantMessage: message({ id: 11 }),
  };
  const textBlock: Block = { id: "b1", kind: "text", seq: 0, text: null, durationMs: null };
  const plan: Plan = {
    phase: "execute",
    status: "running",
    tasks: [{ id: "t1", description: "步骤", dependsOn: ["t0"], status: "pending" }],
    revisions: [
      { revisionNo: 1, reason: "子任务失败", actor: "agent", at: "2026-10-21T10:00:00Z" },
    ],
    awaitingApproval: false,
  };
  const interrupt: Interrupt = {
    id: "it-1",
    type: "confirm",
    confirmKind: "tool_permission",
    action: "shell_exec",
    createdAt: "2026-10-21T10:00:00Z",
    expiresAt: "2026-10-21T10:05:00Z",
    actionable: true,
  };

  it("按 event 类型分发到对应回调并解析 payload", async () => {
    const start = vi.fn();
    const blockStart = vi.fn();
    const blockDelta = vi.fn();
    const blockStop = vi.fn();
    const planUpdate = vi.fn();
    const onInterrupt = vi.fn();
    const onUsage = vi.fn();
    const suggestions = vi.fn();
    const error = vi.fn();
    const end = vi.fn();

    const raw = [
      frame("1", "message.start", startPayload),
      frame("2", "block.start", { messageId: 11, block: textBlock } satisfies BlockStartEvent),
      deltaFrame("3", "b1", "你好"),
      frame("4", "block.stop", {
        messageId: 11,
        block: { ...textBlock, text: "你好", durationMs: 12 },
      } satisfies BlockStopEvent),
      frame("5", "plan.update", { messageId: 11, plan } satisfies PlanUpdateEvent),
      frame("6", "interrupt", { messageId: 11, interrupt }),
      frame("7", "usage", { messageId: 11, usage }),
      frame("8", "suggestions", { messageId: 11, suggestions: ["然后呢"] }),
      frame("9", "error", { code: "B0001", message: "模型超载", messageId: 11 }),
      frame("10", "message.end", {
        messageId: 11,
        status: "failed",
        error: "模型超载",
        usage,
      } satisfies MessageEndEvent),
    ].join("");

    await stream(raw, {
      onStart: start,
      onBlockStart: blockStart,
      onBlockDelta: blockDelta,
      onBlockStop: blockStop,
      onPlanUpdate: planUpdate,
      onInterrupt,
      onUsage,
      onSuggestions: suggestions,
      onError: error,
      onEnd: end,
    });

    expect(start).toHaveBeenCalledWith(startPayload);
    expect(blockStart).toHaveBeenCalledWith({ messageId: 11, block: textBlock });
    expect(blockDelta).toHaveBeenCalledWith({ messageId: 11, blockId: "b1", delta: "你好" });
    expect(blockStop).toHaveBeenCalledWith({
      messageId: 11,
      block: { ...textBlock, text: "你好", durationMs: 12 },
    });
    expect(planUpdate).toHaveBeenCalledWith({ messageId: 11, plan });
    expect(onInterrupt).toHaveBeenCalledWith({ messageId: 11, interrupt });
    expect(onUsage).toHaveBeenCalledWith({ messageId: 11, usage });
    expect(suggestions).toHaveBeenCalledWith({ messageId: 11, suggestions: ["然后呢"] });
    expect(error).toHaveBeenCalledWith({ code: "B0001", message: "模型超载", messageId: 11 });
    expect(end).toHaveBeenCalledWith({
      messageId: 11,
      status: "failed",
      error: "模型超载",
      usage,
    });
  });

  it("三种 block（text/thinking/tool_use）的 start/delta/stop 原样透传", async () => {
    const blocks: Block[] = [
      { id: "t1", kind: "text", seq: 0, text: null, durationMs: null },
      { id: "th1", kind: "thinking", seq: 0, text: null, durationMs: null },
      {
        id: "tu1",
        kind: "tool_use",
        seq: 0,
        text: null,
        durationMs: null,
        tool: {
          callId: "call-1",
          name: "dehaze",
          input: { url: "a.png" },
          status: "running",
          subAgent: false,
        },
      },
    ];
    const startKinds: string[] = [];
    const deltas: BlockDeltaEvent[] = [];
    const stopped: Block[] = [];

    const raw = blocks
      .map(
        (block) =>
          frame(null, "block.start", { messageId: 11, block }) +
          frame(null, "block.delta", { messageId: 11, blockId: block.id, delta: block.kind }) +
          frame(null, "block.stop", {
            messageId: 11,
            block: { ...block, text: block.kind === "tool_use" ? null : "done", durationMs: 5 },
          })
      )
      .join("");

    await stream(raw, {
      onBlockStart: (data) => startKinds.push(data.block.kind),
      onBlockDelta: (data) => deltas.push(data),
      onBlockStop: (data) => stopped.push(data.block),
    });

    expect(startKinds).toEqual(["text", "thinking", "tool_use"]);
    expect(deltas.map((d) => d.blockId)).toEqual(["t1", "th1", "tu1"]);
    expect(stopped.map((b) => b.kind)).toEqual(["text", "thinking", "tool_use"]);
    expect(stopped[2]?.tool?.callId).toBe("call-1");
  });

  it("未知事件名与旧事件名（content_block.* / thought）不触发任何回调", async () => {
    const delta = vi.fn();
    const blockStart = vi.fn();
    const end = vi.fn();
    await stream(
      frame("1", "content_block.delta", { index: 0, delta: { type: "text_delta", text: "旧" } }) +
        frame("2", "content_block.start", { index: 0, type: "text" }) +
        frame("3", "thought", { position: 1, thought: "旧" }) +
        frame("4", "未来新增事件", { a: 1 }),
      { onBlockDelta: delta, onBlockStart: blockStart, onEnd: end }
    );
    expect(delta).not.toHaveBeenCalled();
    expect(blockStart).not.toHaveBeenCalled();
    expect(end).not.toHaveBeenCalled();
  });

  it("ping 仅触发 onPing", async () => {
    const ping = vi.fn();
    const delta = vi.fn();
    await stream(frame("1", "ping", {}) + deltaFrame("2", "b1", "A"), {
      onPing: ping,
      onBlockDelta: delta,
    });
    expect(ping).toHaveBeenCalledTimes(1);
    expect(delta).toHaveBeenCalledTimes(1);
  });
});

describe("终态唯一", () => {
  const end: MessageEndEvent = { messageId: 11, status: "completed", usage };

  it("重复的 message.end 只交付一次", async () => {
    const onEnd = vi.fn();
    await stream(
      frame("1", "message.end", end) + frame("2", "message.end", { ...end, status: "failed" }),
      { onEnd }
    );
    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(onEnd).toHaveBeenCalledWith(end);
  });

  it("message.end 之后的帧不再分发（重放重叠不污染投影）", async () => {
    const onEnd = vi.fn();
    const delta = vi.fn();
    const ping = vi.fn();
    await stream(
      frame("1", "message.end", end) + deltaFrame("2", "b1", "迟到") + frame("3", "ping", {}),
      { onEnd, onBlockDelta: delta, onPing: ping }
    );
    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(delta).not.toHaveBeenCalled();
    expect(ping).not.toHaveBeenCalled();
  });

  it("error 不是终态，其后仍会送达 message.end", async () => {
    const onError = vi.fn();
    const onEnd = vi.fn();
    await stream(
      frame("1", "error", { code: "B0001", message: "模型超载" }) + frame("2", "message.end", end),
      { onError, onEnd }
    );
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onEnd).toHaveBeenCalledTimes(1);
  });

  it("每条流独立计终态（重连新建流可再次收到 message.end）", async () => {
    const first = vi.fn();
    const second = vi.fn();
    await stream(frame("1", "message.end", end), { onEnd: first });
    await stream(frame("1", "message.end", end), { onEnd: second });
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });
});

describe("SSE 传输注入与直返分支", () => {
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

describe("断线重连 Last-Event-ID 请求头", () => {
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

  it("重连流同样按新事件名分发", async () => {
    const onStart = vi.fn();
    const fetchImpl = (async () =>
      sseResponse(
        frame("1", "message.start", {
          streamSessionId: "stream-1",
          userMessage: message({ id: 10, role: "user" }),
          assistantMessage: message(),
        })
      )) as typeof fetch;
    await new Promise<void>((resolve) => {
      AiConversationAPI.reconnectStream(
        1,
        "stream-1",
        "0",
        { onStart, onClose: () => resolve() },
        { fetchImpl }
      );
    });
    expect(onStart).toHaveBeenCalledTimes(1);
  });

  it("返回本次重连的 AbortController，abort 可中断该连接", async () => {
    let receivedSignal: AbortSignal | undefined;
    const onClose = vi.fn();
    const fetchImpl = ((_url: RequestInfo | URL, init?: RequestInit) => {
      receivedSignal = init?.signal ?? undefined;
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
      });
    }) as typeof fetch;

    const controller = AiConversationAPI.reconnectStream(
      1,
      "stream-1",
      "7",
      { onClose },
      { fetchImpl }
    );
    await Promise.resolve();

    expect(controller).toBeInstanceOf(AbortController);
    expect(receivedSignal).toBe(controller.signal);
    expect(controller.signal.aborted).toBe(false);

    controller.abort();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(controller.signal.aborted).toBe(true);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("resume 上行请求体（契约 MessageResume 原样上行）", () => {
  function resume(data: MessageResume) {
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

  it("planEdit 以 camelCase 上行，内层键保持 camelCase", async () => {
    const planEdit = { add: { description: "新增任务", dependsOn: ["t0"], paradigm: "react" } };
    const { fetchMock, done } = resume({ planEdit });
    await done;

    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(String(url).endsWith("/api/v1/ai/messages/7/resume")).toBe(true);
    expect(init?.method).toBe("POST");
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(body).toEqual({ planEdit });
    const raw = JSON.stringify(body);
    expect(raw).not.toContain("plan_edit");
    expect(raw).not.toContain("depends_on");
  });

  it("仅 confirm/params 时只上行这两个键", async () => {
    const { fetchMock, done } = resume({ confirm: true, params: { modelId: "aecrnet" } });
    await done;
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body)) as Record<string, unknown>;
    expect(body).toEqual({ confirm: true, params: { modelId: "aecrnet" } });
  });
});

describe("SSE HTTP 失败携带业务信封", () => {
  function captureError(response: Response): Promise<unknown> {
    const fetchImpl = (async () => response) as typeof fetch;
    return new Promise((resolve) => {
      AiConversationAPI.sendMessage(
        1,
        { content: "x" },
        { onNetworkError: (error) => resolve(error) },
        { fetchImpl }
      );
    });
  }

  it("非 2xx 时错误携带 status/code/msg，供上层映射真实文案", async () => {
    const response = {
      ok: false,
      status: 400,
      statusText: "Bad Request",
      text: async () =>
        JSON.stringify({
          code: "A0400",
          msg: "参数 body.content String should have at most 4000 characters",
        }),
    } as unknown as Response;

    const error = (await captureError(response)) as SSEHttpError;

    expect(error).toBeInstanceOf(SSEHttpError);
    expect(error.status).toBe(400);
    expect(error.code).toBe("A0400");
    expect(error.msg).toContain("at most 4000 characters");
  });

  it("非业务信封响应体（网关错误页）不误报业务码，仅暴露状态码", async () => {
    const response = {
      ok: false,
      status: 502,
      statusText: "Bad Gateway",
      text: async () => "<html>502 Bad Gateway</html>",
    } as unknown as Response;

    const error = (await captureError(response)) as SSEHttpError;

    expect(error).toBeInstanceOf(SSEHttpError);
    expect(error.status).toBe(502);
    expect(error.code).toBeUndefined();
    expect(error.msg).toBeUndefined();
  });
});
