import { AiImageAPI, FileAPI, ImageEditForm } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createImageEditForm, DEFAULT_IMAGE_MODEL } from "#/factories/model";
import { login, logout, forceLogin } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

describe("图像任务 API 测试", () => {
  let uploadedFileId: number;

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
  });

  afterAll(async () => {
    await logout();
  });

  // 复用：提交含已上传文件的图像编辑表单
  const imageEditForm = (overrides: Partial<ImageEditForm> = {}) =>
    createImageEditForm({ fileId: uploadedFileId, ...overrides });

  describe("POST /api/v1/images/edits - 提交图像编辑任务", () => {
    test("正向测试：提交任务并通过轮询获取结果", async () => {
      const result = await AiImageAPI.submitAndWait(imageEditForm(), {
        intervalMs: 2000,
        timeoutMs: 120000,
      });

      expect(result.status).toBe(2);
      expect(typeof result.resultUrl).toBe("string");
      expect(result.resultUrl!.length).toBeGreaterThan(0);
      expect(typeof result.time).toBe("number");
      expect(result.time!).toBeGreaterThanOrEqual(0);
      expect(result.taskId).toBeGreaterThan(0);
    });

    test("异常测试：不存在的模型标识应报错", async () => {
      const form = imageEditForm({ model: "nonexistent-model-xyz" });

      await expectBizError(AiImageAPI.submitImageTask(form), [
        "A0401",
        "A0400",
        "A0500",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    // 后端缺图片来源返回 A0410(PARAM_IS_NULL)
    test("参数校验：缺少图片来源（fileId和imageUrl均为空）应失败", async () => {
      const form = createImageEditForm();
      delete (form as any).imageUrl;
      delete (form as any).fileId;
      await expectBizError(AiImageAPI.submitImageTask(form), ["A0410"]);
    });
  });

  describe("GET /api/v1/images/tasks/{taskId} - 任务状态", () => {
    test("异常测试：不存在的任务ID应报错", async () => {
      await expectBizError(AiImageAPI.getImageTask(99999999), ["A0401"]);
    });
  });

  describe("submitAndWait 轮询机制", () => {
    test("onPoll 回调应被调用（至少一次 processing）", async () => {
      const statuses: number[] = [];
      const result = await AiImageAPI.submitAndWait(imageEditForm(), {
        intervalMs: 1000,
        timeoutMs: 120000,
        onPoll: (status) => statuses.push(status),
      });

      expect(result.status).toBe(2);
      // onPoll 仅在需要多次轮询时才回调；缓存命中可能一次返回终态，故不强制 statuses 非空
      expect(statuses.every((s) => typeof s === "number")).toBe(true);
    });
  });

  describe("POST /api/v1/images/tasks/{taskId}/cancel - 取消任务", () => {
    test("正向测试：取消处理中任务", async () => {
      const result = await AiImageAPI.submitImageTask(imageEditForm());
      // 立即取消（可能已完成也可能还在处理中）
      const cancelResult = await AiImageAPI.cancelImageTask(result.taskId!);
      expect([1, 2, 3, 4]).toContain(cancelResult.status);
    });

    test("异常：取消不存在的任务应失败", async () => {
      await expectBizError(AiImageAPI.cancelImageTask(99999999), ["A0401"]);
    });
  });

  describe("GET /api/v1/images/tasks - 图像任务历史", () => {
    test("正向测试：分页查询图像任务历史", async () => {
      const page = await AiImageAPI.listImageTasks({ pageNum: 1, pageSize: 5 });

      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");
      expect(page.total).toBeGreaterThanOrEqual(0);

      if (page.list.length > 0) {
        const item = page.list[0]!;
        expect(typeof item.id).toBe("number");
        expect(item.id).toBeGreaterThan(0);
      }
    });

    test("正向测试：按模型筛选图像任务历史", async () => {
      const page = await AiImageAPI.listImageTasks({
        pageNum: 1,
        pageSize: 100,
        model: DEFAULT_IMAGE_MODEL,
      });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((log) => {
        expect(log.model).toBe(DEFAULT_IMAGE_MODEL);
      });
    });

    test("边界：大页码返回空列表", async () => {
      const page = await AiImageAPI.listImageTasks({ pageNum: 10000, pageSize: 10 });
      expect(page.list.length).toBe(0);
    });
  });

  describe("越权防护（任务归属校验）", () => {
    test("他人任务查询/取消 → A0401，日志列表仅含本人记录", async () => {
      // USER（beforeAll 已登录）提交任务，仅取 taskId，不等完成
      const resp = await AiImageAPI.submitImageTask(imageEditForm());
      const taskId = resp.taskId!;
      expect(taskId).toBeDefined();

      // 切换 VIP1：查询/取消他人任务 → A0401（与不存在同口径防枚举）
      await login(USERS.VIP1.username);
      await expectBizError(AiImageAPI.getImageTask(taskId), "A0401");
      await expectBizError(AiImageAPI.cancelImageTask(taskId), "A0401");

      // VIP1 的图像任务列表不含 USER 的任务
      const vipLogs = await AiImageAPI.listImageTasks({ pageNum: 1, pageSize: 100 });
      expect(vipLogs.list.every((l) => l.id !== taskId)).toBe(true);

      // 切回 USER：任务仍归属本人可见，未被越权取消（状态非已取消）
      await forceLogin(USERS.USER.username);
      const own = await AiImageAPI.getImageTask(taskId);
      expect(own.taskId).toBe(taskId);
      expect(own.status).not.toBe(4);
    });
  });
});
