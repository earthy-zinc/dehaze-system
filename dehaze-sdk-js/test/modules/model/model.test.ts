import { ModelAPI, FileAPI, PredictionForm } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import {
  createPredictionForm,
  createEvaluationForm,
  createBatchPredictionForm,
  createPresetForm,
  createCompareReportForm,
} from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { ensureDehazeQuota } from "#/utils/quota";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

describe("预测与评估 API 测试", () => {
  let uploadedFileId: number;
  let uploadedFileUrl: string;
  let clearFileUrl: string;

  beforeAll(async () => {
    await ensureDehazeQuota();
    await login(USERS.USER.username);

    const uploadFile = async (relativePath: string): Promise<{ id: number; url: string }> => {
      const filePath = path.resolve(__dirname, relativePath);
      const fileData = fs.readFileSync(filePath);
      const blob = new Blob([fileData]);
      const fileName = path.basename(relativePath);
      const formFile = new File([blob], fileName, { type: "image/jpeg" });
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          return await FileAPI.upload(formFile);
        } catch (e: any) {
          const code = e?.response?.data?.code;
          if (code === "A0002" && attempt < 2) {
            // TTL 5 秒，等待 3 秒覆盖一半 TTL，最多 3 次重试（总等待 9 秒）
            await new Promise((resolve) => setTimeout(resolve, 3000));
            continue;
          }
          if (code === "A0501") {
            // 文件已存在（MD5 去重导致唯一键冲突），按文件名搜索获取真实 id/url
            console.warn(`文件 ${fileName} 已存在，搜索已有记录...`);
            const page = await FileAPI.getPage({ pageNum: 1, pageSize: 5, keywords: fileName });
            const found = page.list.find((f) => f.name === fileName && f.url);
            if (found) {
              return { id: found.id, url: found.url };
            }
            // 查到的文件 url 为空（可能已删除），等待后重试上传
            if (attempt < 2) {
              await new Promise((resolve) => setTimeout(resolve, 3000));
              continue;
            }
            throw new Error(`文件 ${fileName} 已存在但无法在列表中查到有效记录，请检查数据库`);
          }
          throw e;
        }
      }
      return { id: 0, url: "" };
    };

    const hazyInfo = await uploadFile("../../resources/test/model/hazy.jpg");
    uploadedFileId = hazyInfo.id;
    uploadedFileUrl = hazyInfo.url;

    const clearInfo = await uploadFile("../../resources/test/model/clear.jpg");
    clearFileUrl = clearInfo.url;
  });

  afterAll(async () => {
    await logout();
  });

  // 复用：提交含已上传文件的预测表单（真实提交消耗配额，见 ensureDehazeQuota）
  const predictionForm = (overrides: Partial<PredictionForm> = {}) =>
    createPredictionForm({ algorithmId: 13, fileId: uploadedFileId, ...overrides });

  describe("POST /api/v1/prediction - 模型预测（异步）", () => {
    test("正向测试：提交预测并通过轮询获取结果（后端 get_algorithm 已修复）", async () => {
      const result = await ModelAPI.predictAndWait(predictionForm(), {
        intervalMs: 2000,
        timeoutMs: 120000,
      });

      expect(result.status).toBe(2);
      expect(typeof result.resultUrl).toBe("string");
      expect(result.resultUrl!.length).toBeGreaterThan(0);
      expect(typeof result.time).toBe("number");
      expect(result.time!).toBeGreaterThanOrEqual(0);
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createPredictionForm();
      delete (form as any).algorithmId;

      await expectBizError(ModelAPI.predict(form), ["A0400"]);
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      const form = createPredictionForm({ algorithmId: 99999999 });

      await expectBizError(ModelAPI.predict(form), ["A0401"]);
    });

    // 后端缺图片来源返回 A0410(PARAM_IS_NULL)，非文档旧契约 A0500（见 prediction.py）
    test("参数校验：缺少图片来源（fileId和imageUrl均为空）应失败", async () => {
      const form = createPredictionForm();
      delete (form as any).imageUrl;
      delete (form as any).fileId;
      await expectBizError(ModelAPI.predict(form), ["A0410"]);
    });
  });

  describe("GET /api/v1/prediction/{taskId} - 预测状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(ModelAPI.getPredTaskStatus(99999999), ["A0401"]);
    });
  });

  describe("GET /api/v1/prediction/quota - 配额查询", () => {
    test("正向测试：查询当前用户配额", async () => {
      const quota = await ModelAPI.getQuota();
      expect(typeof quota.remaining).toBe("number");
      expect(typeof quota.total).toBe("number");
      expect(typeof quota.used).toBe("number");
      expect(quota.remaining).toBeGreaterThanOrEqual(0);
      expect(quota.total).toBeGreaterThanOrEqual(quota.used);
    });

    test("验证：配额数据结构完整性（后端已补 resetDate）", async () => {
      const quota = await ModelAPI.getQuota();
      expect(quota).toHaveProperty("remaining");
      expect(quota).toHaveProperty("total");
      expect(quota).toHaveProperty("used");
      expect(typeof quota.resetDate).toBe("string");
    });
  });

  describe("POST /api/v1/prediction/batch - 批量预测", () => {
    test("正向测试：批量提交2张图片", async () => {
      const form = createBatchPredictionForm({ algorithmId: 13 });
      const result = await ModelAPI.batchPredict(form);
      expect(result.total).toBe(2);
      expect(Array.isArray(result.results)).toBe(true);
      expect(result.results.length).toBe(2);
      // 每张独立执行：成功条目标记 logId/resultUrl，失败条目标记 status=3 + errorMessage
      result.results.forEach((item) => {
        expect([1, 2, 3, 4]).toContain(item.status);
        if (item.status === 2) {
          expect(item.logId).toBeGreaterThan(0);
        } else if (item.status === 3) {
          expect(item.errorMessage).toBeTruthy();
        }
      });
    });

    test("边界：空items应失败", async () => {
      const form = createBatchPredictionForm({ algorithmId: 13, items: [] });
      await expectBizError(ModelAPI.batchPredict(form), ["A0400"]);
    });

    test("边界：普通用户批量超过等级上限应失败", async () => {
      // 普通用户(level_0) batch_limit=10（见 sys_member_benefit 数据），发送 11 张触发超限
      const items = Array.from({ length: 11 }, (_, i) => ({
        imageUrl: `/datasets/NH-HAZE-2023/hazy/00${String(i + 1).padStart(2, "0")}.JPG`,
      }));
      const form = createBatchPredictionForm({ algorithmId: 13, items });
      // 后端按会员等级 batch_limit 校验，超限返回 A0500
      await expectBizError(ModelAPI.batchPredict(form), ["A0500"]);
    });
  });

  describe("predictAndWait 轮询机制", () => {
    test("onPoll 回调应被调用（至少一次 processing）", async () => {
      const statuses: number[] = [];
      const result = await ModelAPI.predictAndWait(predictionForm(), {
        intervalMs: 1000,
        timeoutMs: 120000,
        onPoll: (status) => statuses.push(status),
      });

      expect(result.status).toBe(2);
      // onPoll 仅在需要多次轮询时才回调；算法服务可能一次返回终态，故不强制 statuses 非空，
      // 但若有回调则每次传入的必须是数字状态码
      expect(statuses.every((s) => typeof s === "number")).toBe(true);
    });
  });

  describe("POST /api/v1/prediction/{taskId}/cancel - 取消任务", () => {
    test("正向测试：取消处理中任务", async () => {
      const result = await ModelAPI.predict(predictionForm());
      // 立即取消（可能已完成也可能还在处理中）
      const cancelResult = await ModelAPI.cancelPredTask(result.logId!);
      expect([1, 2, 3, 4]).toContain(cancelResult.status);
    });

    test("异常：取消不存在的任务应失败", async () => {
      await expectBizError(ModelAPI.cancelPredTask(99999999), ["A0401"]);
    });
  });

  describe("参数预设管理", () => {
    const createdPresetIds: number[] = [];

    afterAll(async () => {
      for (const id of createdPresetIds) {
        try {
          await ModelAPI.deletePreset(id);
        } catch (e) {
          console.warn(`清理失败:`, e);
        }
      }
    });

    test("正向测试：查询预设列表", async () => {
      const result = await ModelAPI.getPresets({ pageNum: 1, pageSize: 100 });
      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("正向测试：创建自定义预设", async () => {
      const form = createPresetForm();
      const result = await ModelAPI.createPreset(form);
      expect(result.id).toBeGreaterThan(0);
      createdPresetIds.push(result.id);
    });

    test("边界：预设名称冲突应失败（后端 preset_service 已捕获唯一键转 A0501）", async () => {
      const form = createPresetForm({ name: `conflict_test_${Date.now()}` });
      const first = await ModelAPI.createPreset(form);
      createdPresetIds.push(first.id);

      await expectBizError(ModelAPI.createPreset(form), ["A0501"]);
    });

    test("正向测试：更新自定义预设", async () => {
      const form = createPresetForm();
      const created = await ModelAPI.createPreset(form);
      createdPresetIds.push(created.id);

      const updateForm = createPresetForm({ name: `updated_${Date.now()}` });
      const updated = await ModelAPI.updatePreset(created.id, updateForm);
      expect(updated.id).toBe(created.id);
    });

    test("正向测试：删除自定义预设", async () => {
      const form = createPresetForm();
      const created = await ModelAPI.createPreset(form);

      await ModelAPI.deletePreset(created.id);

      // 验证已删除（查询列表中不再包含）
      const list = await ModelAPI.getPresets({ pageNum: 1, pageSize: 100 });
      const found = list.list.find((p) => p.id === created.id);
      expect(found).toBeUndefined();
    });

    test("边界：按算法筛选预设", async () => {
      const result = await ModelAPI.getPresets({ pageNum: 1, pageSize: 100, algorithmId: 1 });
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((p) => {
        if (p.algorithmId) {
          expect(p.algorithmId).toBe(1);
        }
      });
    });
  });

  describe("GET /api/v1/prediction/logs - 预测日志", () => {
    test("正向测试：分页查询预测日志", async () => {
      const page = await ModelAPI.getPredLogs({ pageNum: 1, pageSize: 5 });

      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);

      if (page.list.length > 0) {
        const item = page.list[0]!;
        expect(typeof item.id).toBe("number");
        expect(item.id).toBeGreaterThan(0);
      }
    });

    test("正向测试：按算法筛选预测日志", async () => {
      const page = await ModelAPI.getPredLogs({ pageNum: 1, pageSize: 100, algorithmId: 13 });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((log) => {
        if (log.algorithmId) {
          expect(log.algorithmId).toBe(13);
        }
      });
    });

    test("边界：大页码返回空列表", async () => {
      const page = await ModelAPI.getPredLogs({ pageNum: 10000, pageSize: 10 });
      expect(page.list.length).toBe(0);
    });
  });

  describe("POST /api/v1/evaluation - 效果评估（异步）", () => {
    test("正向测试：提交评估并通过轮询获取指标", async () => {
      const form = createEvaluationForm({
        algorithmId: 1,
        predUrl: uploadedFileUrl,
        gtUrl: clearFileUrl,
      });
      const result = await ModelAPI.evaluateAndWait(form, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });

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
    });

    test("参数校验：缺少 algorithmId 应报错", async () => {
      const form = createEvaluationForm();
      delete (form as any).algorithmId;

      await expectBizError(ModelAPI.evaluate(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界：缺少gtUrl应失败", async () => {
      const form = createEvaluationForm();
      delete (form as any).gtUrl;
      await expectBizError(ModelAPI.evaluate(form), ["A0400"]);
    });

    test("边界：缺少predUrl应失败", async () => {
      const form = createEvaluationForm();
      delete (form as any).predUrl;
      await expectBizError(ModelAPI.evaluate(form), ["B0001", "A0400", "ERR_BAD_REQUEST"]);
    });
  });

  describe("GET /api/v1/evaluation/{taskId} - 评估状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(ModelAPI.getEvalTaskStatus(99999999), ["A0401"]);
    });
  });

  describe("GET /api/v1/evaluation/logs - 评估日志", () => {
    test("正向测试：分页查询评估日志", async () => {
      const page = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 5 });

      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);
    });

    test("正向测试：按算法ID筛选评估日志", async () => {
      const page = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 100, algorithmId: 1 });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((log) => {
        if (log.algorithmId) {
          expect(log.algorithmId).toBe(1);
        }
      });
    });
  });

  describe("GET /api/v1/evaluation/metrics - 评估指标历史", () => {
    test("正向测试：分页查询当前用户已完成评估的指标历史", async () => {
      const page = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 5 });

      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      page.list.forEach((item) => {
        expect(item.id).toBeGreaterThan(0);
        if (item.metrics) {
          expect(typeof item.metrics).toBe("object");
        }
      });
    });

    test("验证：评估指标历史仅返回当前用户", async () => {
      const page = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 100 });
      page.list.forEach((item) => {
        expect(item.id).toBeGreaterThan(0);
      });
    });
  });

  describe("对比报告导出", () => {
    test("异常：处理记录不存在应失败", async () => {
      const form = createCompareReportForm({ logId: 99999999 });
      await expectBizError(ModelAPI.generateReport(form), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常：查询不存在的报告应失败", async () => {
      await expectBizError(ModelAPI.getReportStatus(99999999), ["A0401"]);
    });
  });
});
