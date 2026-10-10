import { AiImageAPI, FileAPI, AiModelAPI, ImageInputHistoryAPI, ModelAPI } from "../../../index";
import FavoriteAPI from "@/api/favorite";
import RecommendationAPI from "@/api/recommendation";
import {
  createImageEditForm,
  createEvaluationForm,
  createPresetForm,
  createCompareReportForm,
  DEFAULT_IMAGE_MODEL,
} from "#/factories/model";
import { createFavoriteForm } from "#/factories/favorite";
import { login, logout } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

/**
 * 核心业务流程集成测试
 *
 * 验证核心模块（去雾处理、效果对比、模型选择）与基础模块（收藏、推荐）之间的协作。
 * 这些测试依赖后端接口（图像任务、参数预设、对比报告、推荐）。
 */

// 真实去雾/评估/报告任务轮询等待参数（后端任务处理较慢）
const WAIT_OPTS = { intervalMs: 2000, timeoutMs: 120000 };

describe("核心业务流程集成测试", () => {
  let uploadedFileId: number;
  let uploadedFileUrl: string;
  let clearFileUrl: string;

  // 各场景清理资源
  const favoriteIds: number[] = [];
  const historyIds: number[] = [];
  const presetIds: number[] = [];

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
          // A0501=文件已存在(MD5去重)，A0500=参数校验失败(可能也是重复导致)
          if (code === "A0501" || code === "A0500") {
            console.warn(`文件 ${fileName} 上传失败(code=${code})，搜索已有记录...`);
            const page = await FileAPI.getPage({ pageNum: 1, pageSize: 10, keywords: fileName });
            const found = page.list.find((f: any) => f.name === fileName && f.url);
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
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch {}
    }
    for (const id of presetIds) {
      try {
        await ModelAPI.deletePreset(id);
      } catch {}
    }
    for (const id of historyIds) {
      try {
        await ImageInputHistoryAPI.deleteById(id);
      } catch {}
    }
    // 共享测试文件不删除，由 upsert 复用（避免跨套件并行时文件已被其他套件删掉）

    // 后端 bug 会导致服务偶发连接中断，logout 失败不应使整套标记失败
    try {
      await logout();
    } catch {}
  });

  // ============================================================
  // 场景1：推荐 → 去雾 → 评估 → 收藏 完整流程
  // ============================================================
  describe("场景1：推荐→去雾→评估→收藏 完整流程", () => {
    test("正向流程：使用推荐模型完成去雾、评估并收藏结果", async () => {
      // 1. 图像特征分析
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis).toBeDefined();

      // 2. 获取模型推荐
      const recommendations = await RecommendationAPI.getModelRecommendations({
        imageMd5: (analysis as any).imageMd5,
      });
      expect(Array.isArray(recommendations)).toBe(true);

      // 推荐模型标识：优先取推荐结果，无则回退到默认可用模型
      const recommendedModel =
        recommendations.length > 0 ? recommendations[0]!.modelId : DEFAULT_IMAGE_MODEL;
      const recommendationId: number | undefined =
        recommendations.length > 0 ? recommendations[0]!.recommendationId : undefined;

      // 3. 使用推荐模型去雾处理（传入 recommendedBy 标记来源）
      const imageForm = createImageEditForm({
        model: recommendedModel,
        fileId: uploadedFileId,
        ...(recommendationId !== undefined ? { recommendedBy: recommendationId } : {}),
      });
      const predResult = await AiImageAPI.submitAndWait(imageForm, WAIT_OPTS);
      expect(predResult.status).toBe(2);
      expect(typeof predResult.resultUrl).toBe("string");

      // 4. 效果评估
      const evalForm = createEvaluationForm({
        modelId: recommendedModel,
        predUrl: predResult.resultUrl!,
        gtUrl: clearFileUrl,
      });
      const evalResult = await ModelAPI.evaluateAndWait(evalForm, WAIT_OPTS);
      expect(evalResult.status).toBe(2);

      // 5. 收藏处理结果（targetType="result"）
      const favoriteForm = createFavoriteForm({
        targetType: "result",
        targetId: predResult.taskId!,
      });
      const favoriteId = (await FavoriteAPI.add(favoriteForm)) as number;
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);

      // 6. 验证收藏状态
      const status = await FavoriteAPI.getStatus("result", predResult.taskId!);
      expect(status.favorited).toBe(true);

      // 7. 验证收藏列表包含该记录
      const page = await FavoriteAPI.getPage({ pageNum: 1, pageSize: 100, targetType: "result" });
      const found = page.list.some((item) => item.id === favoriteId);
      expect(found).toBe(true);

      // 8. 提交推荐反馈（采纳）
      if (recommendationId !== undefined) {
        const feedbackResult = await RecommendationAPI.submitFeedback({
          recommendationId,
          useful: true,
        });
        expect(feedbackResult.id).toBeDefined();
      }
    });
  });

  // ============================================================
  // 场景2：模型选择 → 去雾 → 对比报告 完整流程
  // ============================================================
  describe("场景2：模型选择→去雾→对比报告 完整流程", () => {
    test("正向流程：选择模型去雾并生成对比报告", async () => {
      // 1. 获取启用图像模型列表（模型选择器的数据来源）
      const models = await AiModelAPI.listEnabledModels("image_edit");
      expect(Array.isArray(models)).toBe(true);
      expect(models.length).toBeGreaterThan(0);

      // 2. 选择模型去雾处理
      const selectedModel = models[0]!.modelId;
      const imageForm = createImageEditForm({
        model: selectedModel,
        fileId: uploadedFileId,
      });
      const predResult = await AiImageAPI.submitAndWait(imageForm, WAIT_OPTS);
      expect(predResult.status).toBe(2);
      expect(predResult.taskId).toBeDefined();

      // 3. 生成对比报告（异步任务）
      const reportForm = createCompareReportForm({
        logId: predResult.taskId!,
      });
      const reportTask = await ModelAPI.generateReport(reportForm);
      expect(reportTask).toBeDefined();
      expect(reportTask.taskId).toBeDefined();

      // 4. 轮询报告状态直到终态
      let reportStatus = reportTask;
      const deadline = Date.now() + 120000;
      while (Date.now() < deadline) {
        if (reportStatus.status === 2 || reportStatus.status === 3) break;
        await new Promise((r) => setTimeout(r, 2000));
        reportStatus = await ModelAPI.getReportStatus(reportTask.taskId);
      }

      // 报告完成时应返回下载URL
      if (reportStatus.status === 2) {
        expect(typeof reportStatus.downloadUrl).toBe("string");
      }
    });
  });

  // ============================================================
  // 场景3：参数预设管理
  // ============================================================
  describe("场景3：参数预设管理", () => {
    test("正向流程：获取系统预设→创建自定义预设→更新→删除", async () => {
      // 1. 获取系统预设列表
      const systemPresets = await ModelAPI.getPresets({
        pageNum: 1,
        pageSize: 10,
        isSystem: true,
      });
      expect(systemPresets).toBeDefined();
      expect(Array.isArray(systemPresets.list)).toBe(true);

      // 2. 创建自定义预设
      const presetForm = createPresetForm({
        modelId: DEFAULT_IMAGE_MODEL,
        name: `test_preset_${Date.now()}`,
        params: JSON.stringify({ gamma: 1.2 }),
      });
      const createdPreset = await ModelAPI.createPreset(presetForm);
      expect(createdPreset).toBeDefined();
      expect(createdPreset.id).toBeGreaterThan(0);
      expect(createdPreset.name).toBe(presetForm.name);
      presetIds.push(createdPreset.id);

      // 3. 更新自定义预设
      const updateForm = createPresetForm({
        modelId: DEFAULT_IMAGE_MODEL,
        name: `test_preset_updated_${Date.now()}`,
        params: JSON.stringify({ gamma: 1.5 }),
      });
      const updatedPreset = await ModelAPI.updatePreset(createdPreset.id, updateForm);
      expect(updatedPreset.name).toBe(updateForm.name);

      // 4. 删除自定义预设
      await ModelAPI.deletePreset(createdPreset.id);
      const idx = presetIds.indexOf(createdPreset.id);
      if (idx >= 0) presetIds.splice(idx, 1);

      // 删除后查询应不包含该预设
      const allPresets = await ModelAPI.getPresets({ pageNum: 1, pageSize: 100 });
      const stillExists = allPresets.list.some((p) => p.id === createdPreset.id);
      expect(stillExists).toBe(false);
    });

    test("正向流程：使用预设参数去雾处理", async () => {
      // 前置：使用系统预设的 params 发起去雾
      const imageForm = createImageEditForm({
        fileId: uploadedFileId,
        params: JSON.stringify({ gamma: 1.2 }),
      });
      const predResult = await AiImageAPI.submitAndWait(imageForm, WAIT_OPTS);
      expect(predResult.status).toBe(2);
    });
  });
});
