import { beforeEach, describe, expect, it, vi } from "vitest";
import AiObservabilityAPI from "@/api/ai-observability";
import type {
  AiObservabilityTimeline,
  AiObservabilityTimelineQuery,
  AiObservabilityTraceItem,
} from "@/api/ai-observability/model";

/**
 * AI 可观测性 API 请求/响应契约单元测试。
 *
 * mock 掉 axios 封装层，锁定三件事：
 * - URL 与方法：路径拼错会静默 404，类型层面无法兜底；
 * - 时间线 billing 事件的 `tokens` 是**短名** `input/output/cached`（service/ai_observability_service.py
 *   事件组装），不是 inputTokens/cachedInputTokens —— SDK 只透传不重映射，字段名写错前端拿到 undefined；
 * - 已删除的查询参数不得回归（`AiObservabilityTimelineQuery` 对齐 python `TimelineQuery` 仅 include）。
 */

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: requestMock }));

describe("AiObservabilityAPI", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("getTraceDetail 请求 GET /api/v1/ai/observability/traces/{traceId}", async () => {
    await AiObservabilityAPI.getTraceDetail("trace-abc-123");
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/observability/traces/trace-abc-123",
      method: "get",
    });
  });

  it("exportConversationTimeline 请求 timeline/export 且响应为 blob", async () => {
    await AiObservabilityAPI.exportConversationTimeline(42);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/observability/conversations/42/timeline/export",
      method: "get",
      responseType: "blob",
    });
  });
});

describe("AiObservabilityAPI 时间线/检索 wire 契约", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("billing 事件 tokens 用短名，tool_exec 子 Agent 标记为数字", async () => {
    const timeline: AiObservabilityTimeline = {
      conversation: { id: 7, title: "去雾方案讨论" },
      rounds: [
        {
          traces: [
            {
              traceId: "trace-1",
              traceType: "conversation",
              status: 1,
              durationMs: 1200,
              events: [
                {
                  kind: "billing",
                  billType: "chat",
                  credits: 2,
                  tokens: { input: 478, output: 24, cached: 0 },
                },
                { kind: "tool_exec", position: 1, tool: "file_read", isSubagent: 1 },
              ],
            },
          ],
        },
      ],
    };
    requestMock.mockResolvedValueOnce(timeline);

    const result = await AiObservabilityAPI.getConversationTimeline(7, { include: "raw" });
    const events = result.rounds[0]!.traces[0]!.events;

    const billing = events.find((e) => e.kind === "billing")!;
    expect(billing.tokens).toEqual({ input: 478, output: 24, cached: 0 });
    expect(billing.tokens!.input).toBe(478);
    expect(billing.tokens!.cached).toBe(0);

    const toolExec = events.find((e) => e.kind === "tool_exec")!;
    expect(typeof toolExec.isSubagent).toBe("number");
    expect(toolExec.isSubagent).toBe(1);
  });

  it("时间线查询仅发送 include（roundsSinceId/roundsLimit 后端未实现）", async () => {
    requestMock.mockResolvedValueOnce({ conversation: { id: 1, title: "x" }, rounds: [] });

    await AiObservabilityAPI.getConversationTimeline(1, { include: "raw" });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/observability/conversations/1/timeline",
      method: "get",
      params: { include: "raw" },
    });

    // 负向守卫（判据在类型层，由 `npx tsc --noEmit` 验证）：分轮参数已从契约移除；
    // 若被重新加回，下面这行会因"未使用的 @ts-expect-error"而报错。
    const legacy: AiObservabilityTimelineQuery = {
      include: "raw",
      // @ts-expect-error 后端 TimelineQuery 无 roundsLimit
      roundsLimit: 20,
    };
    expect(legacy.include).toBe("raw");
  });

  it("检索项透传会话标题（检索行回填）", async () => {
    const item: AiObservabilityTraceItem = {
      traceId: "trace-9",
      conversationId: 3,
      conversationTitle: "去雾算法选型",
      status: 1,
      durationMs: 1200,
      llmCallCount: 2,
      totalTokens: 500,
      promptTokens: 480,
      completionTokens: 20,
      cachedTokens: 0,
      stepCount: 1,
    };
    requestMock.mockResolvedValueOnce({ list: [item], total: 1 });

    const page = await AiObservabilityAPI.getTraces({ pageNum: 1, pageSize: 10 });
    expect(page.list[0]!.conversationTitle).toBe("去雾算法选型");
  });
});
