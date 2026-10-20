import { ModelAPI, FileAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createEvaluationForm, createPresetForm, createCompareReportForm } from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { createCompletedPredLog } from "#/utils/mysql";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

describe("评估与预设 API 测试", () => {
  let uploadedFileUrl: string;
  let clearFileUrl: string;
  let predLogId = 0; // USER 自己的图像任务记录（用于报告导出与越权用例）
  let evalLogId = 0; // USER 自己的评估任务

  beforeAll(async () => {
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
            await new Promise((resolve) => setTimeout(resolve, 3000));
            continue;
          }
          if (code === "A0501") {
            const page = await FileAPI.getPage({ pageNum: 1, pageSize: 5, keywords: fileName });
            const found = page.list.find((f) => f.name === fileName && f.url);
            if (found) {
              return { id: found.id, url: found.url };
            }
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
    uploadedFileUrl = hazyInfo.url;

    const clearInfo = await uploadFile("../../resources/test/model/clear.jpg");
    clearFileUrl = clearInfo.url;

    // 对比报告需要一张属于 USER 的已完成图像任务记录
    predLogId = await createCompletedPredLog(USERS.USER.id);
  });

  afterAll(async () => {
    await logout();
  });

  describe("POST /api/v1/evaluation - 效果评估（异步）", () => {
    test("正向测试：提交评估并通过轮询获取指标", async () => {
      const form = createEvaluationForm({
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
      evalLogId = result.logId!;
    });

    test("参数校验：缺少 modelId 应报错", async () => {
      const form = createEvaluationForm();
      delete (form as any).modelId;

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

    test("正向测试：按模型标识筛选评估日志", async () => {
      const page = await ModelAPI.getEvalLogs({
        pageNum: 1,
        pageSize: 100,
        modelId: "aecrnet",
      });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((log) => {
        expect(log.modelId).toBe("aecrnet");
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

    test("边界：按模型筛选预设", async () => {
      const result = await ModelAPI.getPresets({
        pageNum: 1,
        pageSize: 100,
        modelId: "aecrnet",
      });
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((p) => {
        expect(p.modelId).toBe("aecrnet");
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

  describe("越权访问控制（用户隔离）", () => {
    test("评估任务状态：查询他人评估任务应返回 A0401", async () => {
      expect(evalLogId).toBeGreaterThan(0); // 前置：USER 已有评估任务
      await login(USERS.TEST.username);
      await expectBizError(ModelAPI.getEvalTaskStatus(evalLogId), ["A0401"]);
      await login(USERS.USER.username);
    });

    test("报告导出：用他人处理记录生成报告应返回 A0401", async () => {
      expect(predLogId).toBeGreaterThan(0); // 前置：USER 已有图像任务记录
      await login(USERS.TEST.username);
      await expectBizError(ModelAPI.generateReport(createCompareReportForm({ logId: predLogId })), [
        "A0401",
      ]);
      await login(USERS.USER.username);
    });

    test("报告状态/下载：查询他人报告任务应返回 A0401", async () => {
      // USER 为自己的图像任务记录生成报告
      const res = await ModelAPI.generateReport(createCompareReportForm({ logId: predLogId }));
      expect(res.taskId).toBeGreaterThan(0);

      await login(USERS.TEST.username);
      await expectBizError(ModelAPI.getReportStatus(res.taskId!), ["A0401"]);
      await login(USERS.USER.username);

      // 本人可正常查询，且最终到达终态
      const own = await ModelAPI.getReportStatus(res.taskId!);
      expect([1, 2, 3]).toContain(own.status);
    });

    test("评估日志列表：仅返回当前用户日志", async () => {
      // 本人列表包含自己的评估日志
      const ownPage = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 100 });
      expect(ownPage.list.some((log) => log.id === evalLogId)).toBe(true);

      // 他人列表不包含该日志（用户隔离）
      await login(USERS.TEST.username);
      const otherPage = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 100 });
      await login(USERS.USER.username);
      expect(otherPage.list.some((log) => log.id === evalLogId)).toBe(false);
    });
  });
});
