import { beforeEach, describe, expect, it, vi } from "vitest";
import AiAgentAPI from "@/api/ai-agent";
import AiEvalAPI from "@/api/ai-eval";

/**
 * 智能体评测与评测中心 API 请求契约单元测试。
 *
 * mock 掉 axios 封装层，锁定 URL 与 wire 字段名（纯 BaseModel 的后端入参不做
 * camelCase 转换：路径拼错 404；字段名拼错时必填项缺参 → 400 + A0400、可选项被
 * 静默忽略——后端校验异常统一 400，不存在 422 响应路径）：
 * - 异步评测任务：POST /runs 返回 task_id，进度走 GET /tasks/{taskId}
 * - 复核详情实为 run+sample 复合定位（/runs/{runId}/samples/{sampleId}），
 *   不是 /reviews/{reviewId}/detail
 */

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: requestMock }));

describe("AiAgentAPI 评测端点", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("runEvalAsync 请求 POST /api/v1/ai/agents/{agentId}/eval/runs", async () => {
    await AiAgentAPI.runEvalAsync(7);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/eval/runs",
      method: "post",
    });
  });

  it("getEvalTask 请求 GET /api/v1/ai/agents/{agentId}/eval/tasks/{taskId}", async () => {
    await AiAgentAPI.getEvalTask(7, "task-abc");
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/eval/tasks/task-abc",
      method: "get",
    });
  });

  it("createEvalDataset 提交 snake_case 的 dataset_type", async () => {
    await AiAgentAPI.createEvalDataset(7, {
      name: "回归集",
      description: "desc",
      datasetType: "regression",
    });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/eval/datasets",
      method: "post",
      data: { name: "回归集", description: "desc", dataset_type: "regression" },
    });
  });

  it("createEvalSample 提交 snake_case 样本字段", async () => {
    await AiAgentAPI.createEvalSample(7, 3, {
      datasetId: 3,
      taskGoal: "目标",
      allowedInput: "text",
      tools: ["file_read"],
      expectedProcess: "过程",
      expectedResult: "结果",
      forbiddenBehavior: "禁止",
      riskLevel: "high",
    });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/eval/datasets/3/samples",
      method: "post",
      data: {
        dataset_id: 3,
        task_goal: "目标",
        allowed_input: "text",
        tools: ["file_read"],
        expected_process: "过程",
        expected_result: "结果",
        forbidden_behavior: "禁止",
        risk_level: "high",
      },
    });
  });

  it("updateEvalSample 请求 PATCH /samples/{sampleId}", async () => {
    await AiAgentAPI.updateEvalSample(7, 9, { taskGoal: "目标2" });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/eval/samples/9",
      method: "patch",
      data: {
        task_goal: "目标2",
        allowed_input: undefined,
        tools: undefined,
        expected_process: undefined,
        expected_result: undefined,
        forbidden_behavior: undefined,
        risk_level: undefined,
      },
    });
  });

  it("publish 提交 change_note 与 force 豁免标识", async () => {
    await AiAgentAPI.publish(7, { changeNote: "变更", force: true });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/7/publish",
      method: "post",
      data: { change_note: "变更", force: true },
    });
  });
});

describe("AiEvalAPI 复核端点", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("getReviewDetail 请求 GET /api/v1/ai/eval-center/runs/{runId}/samples/{sampleId}", async () => {
    await AiEvalAPI.getReviewDetail(12, 34);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/eval-center/runs/12/samples/34",
      method: "get",
    });
  });

  it("submitReview 请求 POST /api/v1/ai/eval-center/reviews/{reviewId}", async () => {
    await AiEvalAPI.submitReview(5, { agree: false, remark: "判分有误" });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/eval-center/reviews/5",
      method: "post",
      data: { agree: false, remark: "判分有误" },
    });
  });
});
