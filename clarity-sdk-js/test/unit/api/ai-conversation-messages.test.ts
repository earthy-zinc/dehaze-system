import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";
import type {
  Interrupt,
  Message,
  MessageList,
  MessageListQuery,
  MessageResume,
  MessageSend,
  Usage,
} from "@/api/ai-conversation/model";

/**
 * 消息读模型与中断查询契约单元测试。
 *
 * mock 掉 axios 封装层，锁定：
 * - 历史消息直接返回契约 `Message` / `MessageList`（含 list/total/hasMore），不再二次改名；
 * - 消息不再有 content/toolCalls/thoughts，正文与推理过程全在 blocks；
 * - 待处理中断可单独查询（进入会话时补查，SSE 中断事件只在推流当下出现）。
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

describe("AiConversationAPI 消息读模型", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ list: [], total: 0, hasMore: false });
  });

  it("getMessages 透传 before/limit/view 至 GET /conversations/{id}/messages", async () => {
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

  it("getMessages 返回契约 MessageList（编译期断言）", () => {
    expectTypeOf(AiConversationAPI.getMessages).returns.toEqualTypeOf<Promise<MessageList>>();
    expectTypeOf<MessageList["list"]>().toEqualTypeOf<Message[]>();
    expectTypeOf<MessageList["hasMore"]>().toEqualTypeOf<boolean>();
    expectTypeOf<MessageList["total"]>().toEqualTypeOf<number>();
  });

  it("getMessageDetail 返回完整 Message", () => {
    expectTypeOf(AiConversationAPI.getMessageDetail).returns.toEqualTypeOf<Promise<Message>>();
    expectTypeOf(AiConversationAPI.getBranches).returns.toEqualTypeOf<Promise<Message[]>>();
    expectTypeOf(AiConversationAPI.stopMessage).returns.toEqualTypeOf<Promise<Message>>();
  });

  it("查询参数含 before/limit/view，不含旧的 pageNum/pageSize", () => {
    expectTypeOf<MessageListQuery>().toHaveProperty("before");
    expectTypeOf<MessageListQuery>().toHaveProperty("limit");
    expectTypeOf<MessageListQuery>().toHaveProperty("view");
    expectTypeOf<MessageListQuery>().not.toHaveProperty("pageNum");
    expectTypeOf<MessageListQuery>().not.toHaveProperty("pageSize");
    expectTypeOf<MessageListQuery["view"]>().toEqualTypeOf<"admin" | undefined>();
  });
});

describe("AiConversationAPI.getMessageInterrupt", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue(null);
  });

  it("请求 GET /api/v1/ai/messages/{id}/interrupt", async () => {
    await AiConversationAPI.getMessageInterrupt(42);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/messages/42/interrupt",
      method: "get",
    });
  });

  it("返回 Interrupt | null（无待处理中断为 null）", () => {
    expectTypeOf(AiConversationAPI.getMessageInterrupt).returns.toEqualTypeOf<
      Promise<Interrupt | null>
    >();
    expectTypeOf<Interrupt["actionable"]>().toEqualTypeOf<boolean>();
    expectTypeOf<Interrupt["expiresAt"]>().toEqualTypeOf<string>();
    expectTypeOf<Interrupt["type"]>().toEqualTypeOf<
      "confirm" | "quota" | "async_wait" | "plan_approve"
    >();
  });
});

describe("Message 形状（新旧字段边界，编译期断言）", () => {
  it("正文/推理/工具全在 blocks，不含 content/toolCalls/thoughts", () => {
    expectTypeOf<Message>().toHaveProperty("blocks");
    expectTypeOf<Message>().toHaveProperty("pendingInterrupt");
    expectTypeOf<Message>().toHaveProperty("plan");
    expectTypeOf<Message>().toHaveProperty("usage");
    expectTypeOf<Message>().toHaveProperty("status");

    expectTypeOf<Message>().not.toHaveProperty("content");
    expectTypeOf<Message>().not.toHaveProperty("toolCalls");
    expectTypeOf<Message>().not.toHaveProperty("thoughts");
    expectTypeOf<Message>().not.toHaveProperty("metadata");
    expectTypeOf<Message>().not.toHaveProperty("toolCallId");
    expectTypeOf<Message>().not.toHaveProperty("originalContent");
  });

  it("status 为字符串通道枚举，不再是数字状态码", () => {
    expectTypeOf<Message["status"]>().toEqualTypeOf<
      "running" | "completed" | "failed" | "canceled" | "awaiting_input"
    >();
  });

  it("请求类型为契约 MessageSend / MessageResume", () => {
    expectTypeOf<MessageSend>().toHaveProperty("content");
    expectTypeOf<MessageSend>().toHaveProperty("model");
    expectTypeOf<MessageSend>().not.toHaveProperty("conversationId");
    expectTypeOf<MessageResume>().toHaveProperty("confirm");
    expectTypeOf<MessageResume>().toHaveProperty("params");
    expectTypeOf<MessageResume>().toHaveProperty("planEdit");
    expectTypeOf<MessageResume>().not.toHaveProperty("plan_edit");
  });
});

describe("Usage 子智能体粒度用量", () => {
  it("subAgents 为可选字段且形状固定（后端不下发时该键不存在）", () => {
    expectTypeOf<Usage["subAgents"]>().toEqualTypeOf<
      | Array<{
          agentCode: string;
          inputTokens: number;
          outputTokens: number;
          cachedInputTokens: number;
          credits: number;
        }>
      | null
      | undefined
    >();

    const withoutSub = {
      inputTokens: 1,
      outputTokens: 2,
      cachedInputTokens: 0,
      credits: 3,
    } satisfies Usage;
    expect(withoutSub).not.toHaveProperty("subAgents");

    const withSub: Usage = {
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
