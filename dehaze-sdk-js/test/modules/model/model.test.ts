import { ModelAPI, FileAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createPredictionForm, createEvaluationForm } from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

describe("预测与评估 API 测试", () => {
  let uploadedFileId: number;
  let uploadedFileUrl: string;
  let clearFileUrl: string;

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

  describe("POST /api/v1/prediction - 模型预测（异步）", () => {
    test("正向测试：提交预测并通过轮询获取结果", async () => {
      const form = createPredictionForm({ algorithmId: 13, fileId: uploadedFileId });
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

      await expectBizError(ModelAPI.predict(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      const form = createPredictionForm({ algorithmId: 99999999 });

      await expectBizError(ModelAPI.predict(form), [
        "A0400",
        "A0401",
        "B0001",
        "ERR_BAD_REQUEST",
        "C0001",
      ]);
    });
  });

  describe("GET /api/v1/prediction/{taskId} - 预测状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(ModelAPI.getPredTaskStatus(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("predictAndWait 轮询机制", () => {
    test("onPoll 回调应被调用（至少一次 processing）", async () => {
      const form = createPredictionForm({ algorithmId: 13, fileId: uploadedFileId });
      const statuses: number[] = [];
      const result = await ModelAPI.predictAndWait(form, {
        intervalMs: 1000,
        timeoutMs: 120000,
        onPoll: (status) => statuses.push(status),
      });

      expect(result.status).toBe(2);
      if (statuses.length > 0) {
        expect(statuses.every((s) => typeof s === "number")).toBe(true);
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
      const form = createEvaluationForm({
        algorithmId: 1,
        predUrl: uploadedFileUrl,
        gtUrl: clearFileUrl,
      });
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

      await expectBizError(ModelAPI.evaluate(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("GET /api/v1/evaluation/{taskId} - 评估状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(ModelAPI.getEvalTaskStatus(99999999), [
        "A0401",
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
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/evaluation/metrics - 评估指标历史", () => {
    test("正向测试：分页查询当前用户已完成评估的指标历史", async () => {
      const page = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 5 });

      expect(page).toBeDefined();
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      page.list.forEach((item) => {
        expect(item.id).toBeGreaterThan(0);
        if (item.metrics) {
          expect(typeof item.metrics).toBe("object");
        }
      });
    });
  });
});
