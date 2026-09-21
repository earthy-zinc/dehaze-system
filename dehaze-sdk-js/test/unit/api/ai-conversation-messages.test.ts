import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";
import type { AiMessageVO, TokenUsage } from "@/api/ai-conversation/model";
import type { CursorResult } from "@/types";

/**
 * 消息列表游标分页契约 + message.end 子智能体粒度用量。
 *
 * mock 掉 axios 封装层，锁定：
 * - getMessages 由 pageNum/pageSize 改为 before/limit（保留 view=admin），返回
 *   `CursorResult<AiMessageVO[]>`（list/total/hasMore），不再暴露 PageResult；
 * - message.end 的 TokenUsage 新增可选 subAgents（仅在存在子智能体调用时下发）。
 */

const { requestMock, serviceMock } = vi.hoisted(() => ({
  requestMock: vi.fn(),
  // ai-conversation 在模块加载期读取 service.defaults.*（SSE 直连复用 axios 配置），mock 需同形状
  serviceMock: { defaults: {} as Record<string, unknown> },
}));

vi.mock("@/utils/request", () => ({
  default: requestMock,
  service: serviceMock,
}));

type MessageCursorQuery = Parameters<typeof AiConversationAPI.getMessages>[1];

describe("AiConversationAPI.getMessages 游标分页契约", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ list: [], total: 0, hasMore: false });
  });

  it("透传 before/limit/view 至 GET /conversations/{id}/messages", async () => {
    await AiConversationAPI.getMessages(7, { before: 100, limit: 20, view: "admin" });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/conversations/7/messages",
      method: "get",
      params: { before: 100, limit: 20, view: "admin" },
    });
  });

  it("省略查询参数时 params 为 undefined（默认 limit=50 由后端兜底）", async () => {
    await AiConversationAPI.getMessages(7);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/conversations/7/messages",
      method: "get",
      params: undefined,
    });
  });

  it("返回 CursorResult<AiMessageVO[]>（编译期断言）", () => {
    expectTypeOf(AiConversationAPI.getMessages).returns.toEqualTypeOf<
      Promise<CursorResult<AiMessageVO[]>>
    >();
    expectTypeOf<CursorResult<AiMessageVO[]>["list"]>().toEqualTypeOf<AiMessageVO[]>();
    expectTypeOf<CursorResult<AiMessageVO[]>["hasMore"]>().toEqualTypeOf<boolean>();
  });

  it("查询参数含 before/limit/view，不含旧的 pageNum/pageSize", () => {
    expectTypeOf<NonNullable<MessageCursorQuery>>().toHaveProperty("before");
    expectTypeOf<NonNullable<MessageCursorQuery>>().toHaveProperty("limit");
    expectTypeOf<NonNullable<MessageCursorQuery>>().toHaveProperty("view");
    expectTypeOf<NonNullable<MessageCursorQuery>>().not.toHaveProperty("pageNum");
    expectTypeOf<NonNullable<MessageCursorQuery>>().not.toHaveProperty("pageSize");
    expectTypeOf<NonNullable<MessageCursorQuery>["view"]>().toEqualTypeOf<"admin" | undefined>();
  });
});

describe("TokenUsage 子智能体粒度用量", () => {
  it("subAgents 为可选字段且形状固定（后端不下发时该键不存在）", () => {
    expectTypeOf<TokenUsage["subAgents"]>().toEqualTypeOf<
      | Array<{
          agentCode: string;
          inputTokens: number;
          outputTokens: number;
          cachedInputTokens: number;
          credits: number;
        }>
      | undefined
    >();

    const withoutSub = {
      inputTokens: 1,
      outputTokens: 2,
      cachedInputTokens: 0,
      credits: 3,
    } satisfies TokenUsage;
    expect(withoutSub).not.toHaveProperty("subAgents");

    const withSub: TokenUsage = {
      ...withoutSub,
      subAgents: [
        {
          agentCode: "sub_a",
          inputTokens: 1,
          outputTokens: 1,
          cachedInputTokens: 0,
          credits: 1,
        },
      ],
    };
    expect(withSub.subAgents?.[0]?.agentCode).toBe("sub_a");
  });
});
