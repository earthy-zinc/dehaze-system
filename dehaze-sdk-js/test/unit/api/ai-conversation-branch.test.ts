import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";
import type {
  AiMessageThought,
  AiMessageVO,
  ConversationVO,
  FeedbackVO,
  InterruptData,
  Plan,
  PlanPayload,
  PlanRevision,
  PlanTask,
} from "@/api/ai-conversation/model";

/**
 * 分支端点请求契约 + plan/中断类型统一后的单元测试。
 *
 * mock 掉 axios 封装层，锁定新增的两个分支端点路径与参数拼装（严格对齐 python
 * `GET /conversations/{conv_id}/messages/{msg_id}/branches` 与
 * `PUT /conversations/{conv_id}/branches/{msg_id}`），并以编译期断言锁定：
 * `Plan`（事件完整形状）与 `PlanPayload`（中断形状，仅去掉 `phase`）同形，
 * 中断数据与计划任务项全 camelCase。
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

describe("AiConversationAPI 分支端点", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("getBranches 请求 GET /conversations/{id}/messages/{msgId}/branches", async () => {
    await AiConversationAPI.getBranches(1, 4);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/conversations/1/messages/4/branches",
      method: "get",
    });
  });

  it("switchBranch 请求 PUT /conversations/{id}/branches/{msgId}（无请求体）", async () => {
    await AiConversationAPI.switchBranch(1, 4);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/conversations/1/branches/4",
      method: "put",
    });
  });

  it("getBranches 返回消息列表、switchBranch 返回会话（编译期断言）", () => {
    expectTypeOf(AiConversationAPI.getBranches).returns.toEqualTypeOf<Promise<AiMessageVO[]>>();
    expectTypeOf(AiConversationAPI.switchBranch).returns.toEqualTypeOf<Promise<ConversationVO>>();
  });
});

describe("plan 统一形状（编译期断言）", () => {
  it("PlanPayload 为 Plan 去掉 phase，phase 仅事件路径存在", () => {
    expectTypeOf<PlanPayload>().toEqualTypeOf<Omit<Plan, "phase">>();
    expectTypeOf<Plan["tasks"]>().toEqualTypeOf<PlanTask[]>();
    expectTypeOf<PlanPayload["tasks"]>().toEqualTypeOf<PlanTask[]>();
    expectTypeOf<PlanPayload["revisions"]>().toEqualTypeOf<PlanRevision[]>();
    expectTypeOf<InterruptData["plan"]>().toEqualTypeOf<PlanPayload | undefined>();

    // 事件完整形状可赋给中断形状（多出的 phase 不破坏形状）
    const event: Plan = { tasks: [], revisions: [], phase: "plan" };
    const asPayload: PlanPayload = event;
    expect(asPayload).toBe(event);
    expectTypeOf<Plan["phase"]>().toEqualTypeOf<string | undefined>();
    expectTypeOf<PlanPayload>().not.toHaveProperty("phase");
  });

  it("任务项与修订项为统一 camelCase 形状，不含旧键", () => {
    expectTypeOf<PlanTask>().toHaveProperty("dependsOn");
    expectTypeOf<PlanTask>().not.toHaveProperty("depends_on");
    expectTypeOf<PlanTask>().not.toHaveProperty("tool_hint");

    expectTypeOf<PlanRevision>().toHaveProperty("revisionNo");
    expectTypeOf<PlanRevision>().toHaveProperty("changedTaskIds");
    expectTypeOf<PlanRevision>().not.toHaveProperty("at");
    expectTypeOf<PlanRevision>().not.toHaveProperty("change");
  });

  it("InterruptData 按统一后的 camelCase 键声明，不含旧 snake_case 键", () => {
    expectTypeOf<InterruptData>().toHaveProperty("confirmKind");
    expectTypeOf<InterruptData>().toHaveProperty("previousWriter");
    expectTypeOf<InterruptData>().toHaveProperty("artifactId");
    expectTypeOf<InterruptData>().toHaveProperty("upgradeTip");
    expectTypeOf<InterruptData>().toHaveProperty("usedDaily");
    expectTypeOf<InterruptData>().toHaveProperty("dailyLimit");
    expectTypeOf<InterruptData>().toHaveProperty("usedMonthly");
    expectTypeOf<InterruptData>().toHaveProperty("monthlyLimit");
    expectTypeOf<InterruptData>().toHaveProperty("quotaDataError");
    expectTypeOf<InterruptData>().toHaveProperty("taskId");
    expectTypeOf<InterruptData>().toHaveProperty("taskType");
    expectTypeOf<InterruptData>().toHaveProperty("estDuration");
    expectTypeOf<InterruptData>().toHaveProperty("imageCount");

    expectTypeOf<InterruptData>().not.toHaveProperty("confirm_kind");
    expectTypeOf<InterruptData>().not.toHaveProperty("previous_writer");
    expectTypeOf<InterruptData>().not.toHaveProperty("upgrade_tip");
    expectTypeOf<InterruptData>().not.toHaveProperty("used_daily");
    expectTypeOf<InterruptData>().not.toHaveProperty("daily_limit");
    expectTypeOf<InterruptData>().not.toHaveProperty("used_monthly");
    expectTypeOf<InterruptData>().not.toHaveProperty("monthly_limit");
    expectTypeOf<InterruptData>().not.toHaveProperty("quota_data_error");
    expectTypeOf<InterruptData>().not.toHaveProperty("task_id");
    expectTypeOf<InterruptData>().not.toHaveProperty("task_type");
    expectTypeOf<InterruptData>().not.toHaveProperty("est_duration");
    expectTypeOf<InterruptData>().not.toHaveProperty("image_count");
  });
});

describe("弥补落后于后端的字段（编译期断言）", () => {
  it("思考步骤/消息/反馈新增字段为可选且不带显式 null", () => {
    expectTypeOf<AiMessageThought["agentCode"]>().toEqualTypeOf<string | undefined>();
    expectTypeOf<AiMessageThought["isSubagent"]>().toEqualTypeOf<number | undefined>();
    expectTypeOf<AiMessageVO["metadata"]>().toEqualTypeOf<unknown>();
    expectTypeOf<FeedbackVO["updateTime"]>().toEqualTypeOf<string | undefined>();
  });
});
