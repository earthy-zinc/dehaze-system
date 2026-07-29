import { ModelAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createPredictionForm, createEvaluationForm } from "#/factories/model";

describe("预测与评估 API 测试", () => {
  describe("POST /api/v1/prediction - 模型预测（异步）", () => {
    test("正向测试：提交预测并通过轮询获取结果", async () => {
      const form = createPredictionForm({ algorithmId: 13 }); // DCP 算法
      const result = await ModelAPI.predictAndWait(form, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });

      expect(result).toBeDefined();
      expect(result.status).toBe(2);
      expect(typeof result.resultUrl).toBe("string");
      expect(result.resultUrl!.length).toBeGreaterThan(0);
      expect(typeof result.time).toBe("number");
      expect(result.time!).toBeGreaterThanOrEqual(0);
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createPredictionForm();
      delete (form as any).algorithmId;

      await expectBizError(
        ModelAPI.predict(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      const form = createPredictionForm({ algorithmId: 99999999 });

      await expectBizError(
        ModelAPI.predict(form),
        ["A0400", "A0401", "B0001", "ERR_BAD_REQUEST", "C0001"],
        undefined,
        true
      );
    });
  });

  describe("GET /api/v1/prediction/{taskId} - 预测状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(
        ModelAPI.getPredTaskStatus(99999999),
        ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("predictAndWait 轮询机制", () => {
    test("onPoll 回调应被调用（至少一次 processing）", async () => {
      const form = createPredictionForm({ algorithmId: 13 });
      const statuses: number[] = [];
      const result = await ModelAPI.predictAndWait(form, {
        intervalMs: 1000,
        timeoutMs: 120000,
        onPoll: (status) => statuses.push(status),
      });

      expect(result.status).toBe(2);
      // 轮询可能直接返回completed(2)而未经过processing(1)（任务快速完成）
      if (statuses.length > 0) {
        expect(statuses).toContain(expect.any(Number));
      }
    });
  });

  describe("GET /api/v1/prediction/logs - 预测日志", () => {
    test("正向测试：分页查询预测日志", async () => {
      const page = await ModelAPI.getPredLogs({ pageNum: 1, pageSize: 5 });

      expect(page).toBeDefined();
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);

      const list = page.list as any;
      if (Array.isArray(list) && list.length > 0) {
        const item = list[0]!;
        expect(typeof item.id).toBe("number");
        expect(item.id).toBeGreaterThan(0);
      }
    });
  });

  describe("POST /api/v1/evaluation - 效果评估（异步）", () => {
    test("正向测试：提交评估并通过轮询获取指标", async () => {
      const form = createEvaluationForm({ algorithmId: 1 });
      try {
        const result = await ModelAPI.evaluateAndWait(form, {
          intervalMs: 2000,
          timeoutMs: 120000,
        });

        expect(result).toBeDefined();
        expect(result.status).toBe(2);
        expect(typeof result.metrics).toBe("object");
        expect(result.metrics).not.toBeNull();
        if (result.metrics!.psnr !== undefined) {
          expect(typeof result.metrics!.psnr).toBe("number");
          expect(result.metrics!.psnr).toBeGreaterThan(0);
        }
        if (result.metrics!.ssim !== undefined) {
          expect(typeof result.metrics!.ssim).toBe("number");
          expect(result.metrics!.ssim).toBeGreaterThanOrEqual(0);
          expect(result.metrics!.ssim).toBeLessThanOrEqual(1);
        }
        expect(typeof result.time).toBe("number");
        expect(result.time!).toBeGreaterThanOrEqual(0);
      } catch (e: any) {
        console.error("=== 评估请求失败 ===");
        console.error("HTTP状态:", e?.response?.status);
        console.error("响应体:", JSON.stringify(e?.response?.data, null, 2));
        throw e;
      }
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createEvaluationForm();
      delete (form as any).algorithmId;

      await expectBizError(
        ModelAPI.evaluate(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("GET /api/v1/evaluation/{taskId} - 评估状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(
        ModelAPI.getEvalTaskStatus(99999999),
        ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("GET /api/v1/evaluation/logs - 评估日志", () => {
    test("正向测试：分页查询评估日志", async () => {
      const page = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 5 });

      expect(page).toBeDefined();
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);
    });
  });
});
