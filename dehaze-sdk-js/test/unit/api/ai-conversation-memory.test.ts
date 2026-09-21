import { beforeEach, describe, expect, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";

/**
 * 长期记忆 API 请求契约单元测试。
 *
 * mock 掉 axios 封装层，锁定归档相关两个端点与更新接口的 wire 载荷：
 * - 取消归档是独立子路径（POST /memories/{id}/unarchive，无请求体）而非 PUT 状态字段，
 *   后端已删除 MemoryUpdate.archived 且无手动归档入口，回传 archived 会被静默忽略（非 422）
 * - 归档列表走 /memories/archived（普通列表不含归档记忆）
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

describe("AiConversationAPI 记忆归档端点", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("getArchivedMemories 请求 GET /api/v1/ai/memories/archived", async () => {
    await AiConversationAPI.getArchivedMemories({ pageNum: 1, pageSize: 20 });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/memories/archived",
      method: "get",
      params: { pageNum: 1, pageSize: 20 },
    });
  });

  it("unarchiveMemory 请求 POST /api/v1/ai/memories/{id}/unarchive（无请求体）", async () => {
    await AiConversationAPI.unarchiveMemory(7);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/memories/7/unarchive",
      method: "post",
    });
  });

  it("updateMemory 只提交传入字段（不含 archived）", async () => {
    await AiConversationAPI.updateMemory(7, { content: "更新后的内容" });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/memories/7",
      method: "put",
      data: { content: "更新后的内容" },
    });
  });
});
