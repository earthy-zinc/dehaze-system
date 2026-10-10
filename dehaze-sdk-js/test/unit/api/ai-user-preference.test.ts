import { beforeEach, describe, expect, it, vi } from "vitest";
import { AiUserPreferenceAPI } from "@/api/ai-model";
import type { AiUserPreferenceForm } from "@/api/ai-model/model";

/**
 * 用户偏好默认模型 API 请求契约单元测试（mock axios 封装层，锁 URL/wire 字段）。
 *
 * 契约：GET/PUT /api/v1/ai/user-preference，字段名 defaultModel，null 表示清除偏好。
 */

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: requestMock }));

describe("AiUserPreferenceAPI 请求契约", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ defaultModel: null });
  });

  it("get 查询用户偏好默认模型", async () => {
    await AiUserPreferenceAPI.get();
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/user-preference",
      method: "get",
    });
  });

  it("update 以 PUT 提交 defaultModel", async () => {
    const form: AiUserPreferenceForm = { defaultModel: "qwen3-0.6b" };
    await AiUserPreferenceAPI.update(form);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/user-preference",
      method: "put",
      data: form,
    });
  });

  it("update 以 defaultModel: null 清除偏好（回落平台默认模型）", async () => {
    await AiUserPreferenceAPI.update({ defaultModel: null });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/user-preference",
      method: "put",
      data: { defaultModel: null },
    });
  });

  it("defaultModel 为必填字段（清除偏好需显式传 null）", async () => {
    // 判据在类型层，由 `npx tsc --noEmit` 验证：改为可选会使本行报"未使用的 @ts-expect-error"。
    // @ts-expect-error defaultModel 必填
    await AiUserPreferenceAPI.update({});
  });
});
