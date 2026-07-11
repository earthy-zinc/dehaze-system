import { ModelAPI } from "../../../index";
import { login, logout } from "#/utils/auth";
import { expectBizErrorOrUndefined } from "#/utils/assertion";
import { createPredictionForm, createEvaluationForm } from "#/factories/model";

describe("预测与评估 API 测试", () => {
  beforeAll(async () => {
    await login();
  }, 30000);

  afterAll(async () => {
    await logout();
  });

  describe("POST /api/v1/prediction - 模型预测", () => {
    test("正向测试：提交预测请求并验证返回结构", async () => {
      const form = createPredictionForm({ algorithmId: 1 });
      const result = await ModelAPI.predict(form);

      expect(result).toBeDefined();
      expect(result).toHaveProperty("resultUrl");
      expect(result).toHaveProperty("time");
      expect(typeof result.time).toBe("number");
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createPredictionForm();
      delete (form as any).algorithmId;

      await expectBizErrorOrUndefined(ModelAPI.predict(form), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      const form = createPredictionForm({ algorithmId: 99999999 });

      await expectBizErrorOrUndefined(ModelAPI.predict(form), [
        "A0400",
        "A0401",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/prediction/{taskId} - 预测状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizErrorOrUndefined(ModelAPI.getPredTaskStatus(99999999), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/prediction/logs - 预测日志", () => {
    test("正向测试：分页查询预测日志", async () => {
      const page = await ModelAPI.getPredLogs({ pageNum: 1, pageSize: 5 });

      expect(page).toBeDefined();
      expect(page).toHaveProperty("list");
      expect(page).toHaveProperty("total");
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");

      // 验证列表项结构
      if (page.list.length > 0) {
        const item = page.list[0];
        expect(item).toHaveProperty("id");
        expect(item).toHaveProperty("algorithmId");
      }
    });
  });

  describe("POST /api/v1/evaluation - 效果评估", () => {
    test("正向测试：提交评估请求并验证返回指标", async () => {
      const form = createEvaluationForm({ algorithmId: 1 });
      const result = await ModelAPI.evaluate(form);

      expect(result).toBeDefined();
      expect(result).toHaveProperty("metrics");
      expect(result).toHaveProperty("time");
      expect(typeof result.metrics).toBe("object");
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createEvaluationForm();
      delete (form as any).algorithmId;

      await expectBizErrorOrUndefined(ModelAPI.evaluate(form), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/evaluation/{taskId} - 评估状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizErrorOrUndefined(ModelAPI.getEvalTaskStatus(99999999), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/evaluation/logs - 评估日志", () => {
    test("正向测试：分页查询评估日志", async () => {
      const page = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 5 });

      expect(page).toBeDefined();
      expect(page).toHaveProperty("list");
      expect(page).toHaveProperty("total");
      expect(Array.isArray(page.list)).toBe(true);
    });
  });
});
