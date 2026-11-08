import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";
import type { Feedback, Message } from "@/api/ai-conversation/model";

/**
 * 会话级批量反馈查询单元测试。
 *
 * mock 掉 axios 封装层，锁定：
 * - 端点与参数拼装对齐 `GET /conversations/{conv_id}/feedback`（无 query、无 body）；
 * - 无反馈时原样返回空数组，不把 `[]` 视为空值；
 * - 错误原样透传（不在 SDK 内吞掉或改写）；
 * - 返回类型为契约 `Feedback[]`，且 `Message` 已带分支兄弟序号字段。
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

describe("AiConversationAPI.getConversationFeedback", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("请求 GET /api/v1/ai/conversations/{id}/feedback（无参数、无请求体）", async () => {
    requestMock.mockResolvedValue([]);
    await AiConversationAPI.getConversationFeedback(9);
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/conversations/9/feedback",
      method: "get",
    });
  });

  it("无反馈时原样返回空数组（不是 undefined/null）", async () => {
    requestMock.mockResolvedValue([]);
    await expect(AiConversationAPI.getConversationFeedback(9)).resolves.toEqual([]);
  });

  it("错误原样透传（鉴权失败/网络错误不在 SDK 内吞掉）", async () => {
    const error = new Error("A0401 未登录");
    requestMock.mockRejectedValue(error);
    await expect(AiConversationAPI.getConversationFeedback(9)).rejects.toBe(error);
  });

  it("返回契约 Feedback 数组（编译期断言）", () => {
    expectTypeOf(AiConversationAPI.getConversationFeedback).returns.toEqualTypeOf<
      Promise<Feedback[]>
    >();
    expectTypeOf<Feedback["messageId"]>().toEqualTypeOf<number>();
    expectTypeOf<Feedback["rating"]>().toEqualTypeOf<1 | -1>();
    expectTypeOf<Feedback["tags"]>().toEqualTypeOf<string[] | null | undefined>();
  });
});

describe("Message 分支兄弟字段（编译期断言）", () => {
  it("siblingIndex/siblingCount 为可选且可空，旧消息不带该键也合法", () => {
    expectTypeOf<Message>().toHaveProperty("siblingIndex");
    expectTypeOf<Message>().toHaveProperty("siblingCount");
    expectTypeOf<Message["siblingIndex"]>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<Message["siblingCount"]>().toEqualTypeOf<number | null | undefined>();
  });
});
