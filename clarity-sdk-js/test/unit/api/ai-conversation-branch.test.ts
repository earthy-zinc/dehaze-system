import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import AiConversationAPI from "@/api/ai-conversation";
import type {
  ConversationVO,
  Interrupt,
  Message,
  Plan,
  PlanRevision,
  PlanTask,
  ToolCall,
} from "@/api/ai-conversation/model";

/**
 * 分支端点请求契约 + 计划/中断契约形状单元测试。
 *
 * mock 掉 axios 封装层，锁定两个分支端点路径与参数拼装（对齐
 * `GET /conversations/{conv_id}/messages/{msg_id}/branches` 与
 * `PUT /conversations/{conv_id}/branches/{msg_id}`），并以编译期断言锁定契约生成的
 * `Plan` / `Interrupt` 形状：全 camelCase、无旧键。
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
    expectTypeOf(AiConversationAPI.getBranches).returns.toEqualTypeOf<Promise<Message[]>>();
    expectTypeOf(AiConversationAPI.switchBranch).returns.toEqualTypeOf<Promise<ConversationVO>>();
  });
});

describe("Plan 契约形状（编译期断言）", () => {
  it("Plan 必含 status/tasks/revisions/awaitingApproval，phase 可选且仅在事件路径出现", () => {
    expectTypeOf<Plan>().toHaveProperty("status");
    expectTypeOf<Plan>().toHaveProperty("tasks");
    expectTypeOf<Plan>().toHaveProperty("revisions");
    expectTypeOf<Plan>().toHaveProperty("awaitingApproval");
    expectTypeOf<Plan["tasks"]>().toEqualTypeOf<PlanTask[]>();
    expectTypeOf<Plan["revisions"]>().toEqualTypeOf<PlanRevision[]>();
    expectTypeOf<Plan["awaitingApproval"]>().toEqualTypeOf<boolean>();
    expectTypeOf<Plan["status"]>().toEqualTypeOf<"pending" | "running" | "completed" | "failed">();

    // phase 可选：不传也合法
    const plan: Plan = { status: "pending", tasks: [], revisions: [], awaitingApproval: false };
    expect(plan).not.toHaveProperty("phase");
  });

  it("任务项与修订项为契约 camelCase 形状，不含旧键", () => {
    expectTypeOf<PlanTask>().toHaveProperty("dependsOn");
    expectTypeOf<PlanTask>().toHaveProperty("paradigm");
    expectTypeOf<PlanTask>().not.toHaveProperty("depends_on");
    expectTypeOf<PlanTask>().not.toHaveProperty("tool_hint");
    expectTypeOf<PlanTask>().not.toHaveProperty("toolHint");
    expectTypeOf<PlanTask>().not.toHaveProperty("result");

    expectTypeOf<PlanRevision>().toHaveProperty("revisionNo");
    expectTypeOf<PlanRevision>().toHaveProperty("reason");
    expectTypeOf<PlanRevision>().toHaveProperty("actor");
    expectTypeOf<PlanRevision>().toHaveProperty("at");
    expectTypeOf<PlanRevision>().not.toHaveProperty("changedTaskIds");
  });
});

describe("Interrupt 契约形状（编译期断言）", () => {
  it("按契约 camelCase 键声明，含可续期语义，不含旧 snake_case 键", () => {
    expectTypeOf<Interrupt>().toHaveProperty("confirmKind");
    expectTypeOf<Interrupt>().toHaveProperty("action");
    expectTypeOf<Interrupt>().toHaveProperty("recommendation");
    expectTypeOf<Interrupt>().toHaveProperty("alternatives");
    expectTypeOf<Interrupt>().toHaveProperty("upgradeTip");
    expectTypeOf<Interrupt>().toHaveProperty("usedDaily");
    expectTypeOf<Interrupt>().toHaveProperty("dailyLimit");
    expectTypeOf<Interrupt>().toHaveProperty("usedMonthly");
    expectTypeOf<Interrupt>().toHaveProperty("monthlyLimit");
    expectTypeOf<Interrupt>().toHaveProperty("estDuration");
    expectTypeOf<Interrupt>().toHaveProperty("imageCount");
    expectTypeOf<Interrupt>().toHaveProperty("plan");
    expectTypeOf<Interrupt>().toHaveProperty("createdAt");
    expectTypeOf<Interrupt>().toHaveProperty("expiresAt");
    expectTypeOf<Interrupt>().toHaveProperty("actionable");

    expectTypeOf<Interrupt>().not.toHaveProperty("confirm_kind");
    expectTypeOf<Interrupt>().not.toHaveProperty("upgrade_tip");
    expectTypeOf<Interrupt>().not.toHaveProperty("used_daily");
    expectTypeOf<Interrupt>().not.toHaveProperty("daily_limit");
    expectTypeOf<Interrupt>().not.toHaveProperty("used_monthly");
    expectTypeOf<Interrupt>().not.toHaveProperty("monthly_limit");
    expectTypeOf<Interrupt>().not.toHaveProperty("est_duration");
    expectTypeOf<Interrupt>().not.toHaveProperty("image_count");
    expectTypeOf<Interrupt>().not.toHaveProperty("data");
    expectTypeOf<Interrupt>().not.toHaveProperty("artifactId");
  });

  it("工具调用为契约 ToolCall，状态为字符串枚举", () => {
    expectTypeOf<ToolCall>().toHaveProperty("callId");
    expectTypeOf<ToolCall>().toHaveProperty("subAgent");
    expectTypeOf<ToolCall["status"]>().toEqualTypeOf<
      "running" | "success" | "failed" | "skipped"
    >();
    expectTypeOf<ToolCall>().not.toHaveProperty("tool_call_id");
  });
});
