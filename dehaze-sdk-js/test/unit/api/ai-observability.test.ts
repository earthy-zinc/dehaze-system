import { beforeEach, describe, expect, it, vi } from "vitest";
import AiObservabilityAPI from "@/api/ai-observability";

/**
 * AI 可观测性 API 请求契约单元测试。
 *
 * mock 掉 axios 封装层，锁定 getTraceDetail 的请求 URL 与方法——
 * 过程链详情是普通用户唯一可访问的端点，路径拼错会静默 404，类型层面无法兜底。
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
});
