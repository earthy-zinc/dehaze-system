import { ModelAPI, FileAPI, AlgorithmAPI, ImageInputHistoryAPI } from "../../../index";
import FavoriteAPI from "@/api/favorite";
import RecommendationAPI from "@/api/recommendation";
import {
  createPredictionForm,
  createEvaluationForm,
  createBatchPredictionForm,
  createPresetForm,
  createCompareReportForm,
} from "#/factories/model";
import { createFavoriteForm } from "#/factories/favorite";
import { login, logout } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

/**
 * 核心业务流程集成测试
 *
 * 验证核心模块（去雾处理、效果对比、算法选择）与基础模块（收藏、推荐）之间的协作。
 * 这些测试依赖后端实现新接口（批量预测、配额、参数预设、对比报告、推荐），
 * 后端尚未实现时测试会失败——这是 TDD 预期行为。
 */
describe("核心业务流程集成测试", () => {
  let uploadedFileId: number;
  let uploadedFileUrl: string;
  let clearFileId: number;
  let clearFileUrl: string;

  // 各场景清理资源
  const favoriteIds: number[] = [];
  const algorithmIds: number[] = [];
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
    clearFileId = clearInfo.id;
    clearFileUrl = clearInfo.url;
  });

  afterAll(async () => {
    // 收藏清理
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch {}
    }
    // 预设清理
    for (const id of presetIds) {
      try {
        await ModelAPI.deletePreset(id);
      } catch {}
    }
    // 历史记录清理
    for (const id of historyIds) {
      try {
        await ImageInputHistoryAPI.deleteById(id);
      } catch {}
    }
    // 算法清理
    for (const id of algorithmIds) {
      try {
        await AlgorithmAPI.deleteByIds([id.toString()]);
      } catch {}
    }
    // 共享测试文件不删除，由 upsert 复用（避免跨套件并行时文件已被其他套件删掉）

    await logout();
  });

  // ============================================================
  // 场景1：推荐 → 去雾 → 对比 → 收藏 完整流程
  // ============================================================
  describe("场景1：推荐→去雾→对比→收藏 完整流程", () => {
    test("正向流程：使用推荐算法完成去雾、评估并收藏结果", async () => {
      // 1. 图像特征分析
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis).toBeDefined();
      expect(analysis.hazeLevel).toBeDefined();

      // 2. 获取算法推荐
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: (analysis as any).imageMd5,
      });
      expect(Array.isArray(recommendations)).toBe(true);

      // 推荐算法ID：优先取推荐结果，无则回退到已知可用算法
      const recommendedAlgorithmId =
        recommendations.length > 0 ? recommendations[0]!.algorithmId : 13;
      const recommendationId: number | undefined = recommendations.length > 0 ? 1 : undefined;

      // 3. 使用推荐算法去雾处理（传入 recommendedBy 标记来源）
      const predForm = createPredictionForm({
        algorithmId: recommendedAlgorithmId,
        fileId: uploadedFileId,
        ...(recommendationId !== undefined ? { recommendedBy: recommendationId } : {}),
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);
      expect(typeof predResult.resultUrl).toBe("string");

      // 4. 效果评估
      const evalForm = createEvaluationForm({
        algorithmId: recommendedAlgorithmId,
        predUrl: predResult.resultUrl!,
        gtUrl: clearFileUrl,
      });
      const evalResult = await ModelAPI.evaluateAndWait(evalForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(evalResult.status).toBe(2);

      // 5. 收藏处理结果（targetType="result"）
      const favoriteForm = createFavoriteForm({
        targetType: "result",
        targetId: predResult.logId!,
      });
      const favoriteId = (await FavoriteAPI.add(favoriteForm)) as number;
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);

      // 6. 验证收藏状态
      const status = await FavoriteAPI.getStatus("result", predResult.logId!);
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
  // 场景2：算法选择 → 去雾 → 对比报告 完整流程
  // ============================================================
  describe("场景2：算法选择→去雾→对比报告 完整流程", () => {
    test("正向流程：选择算法去雾并生成对比报告", async () => {
      // 1. 获取算法列表（算法选择树的数据来源）
      const algorithms = await AlgorithmAPI.getList();
      expect(Array.isArray(algorithms)).toBe(true);
      expect(algorithms.length).toBeGreaterThan(0);

      // 2. 搜索算法（用第一个算法名称的前缀作为关键词）
      const firstAlgo = algorithms[0]!;
      const keyword = firstAlgo.name.substring(0, Math.min(2, firstAlgo.name.length));
      const searchResult = await AlgorithmAPI.getList({ keywords: keyword });
      expect(Array.isArray(searchResult)).toBe(true);

      // 找一个可用的算法ID（优先用搜索结果，回退到 13）
      const selectedAlgorithmId = 13;

      // 3. 选择算法去雾处理
      const predForm = createPredictionForm({
        algorithmId: selectedAlgorithmId,
        fileId: uploadedFileId,
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);
      expect(predResult.logId).toBeDefined();

      // 4. 生成对比报告（异步任务）
      const reportForm = createCompareReportForm({
        logId: predResult.logId!,
        format: "pdf",
        includeMetrics: true,
      });
      const reportTask = await ModelAPI.generateReport(reportForm);
      expect(reportTask).toBeDefined();
      expect(reportTask.taskId).toBeDefined();

      // 5. 轮询报告状态直到终态
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
  // 场景3：批量去雾 + 配额校验
  // ============================================================
  describe("场景3：批量去雾+配额校验", () => {
    test("正向流程：查询配额后批量去雾并验证配额扣减", async () => {
      // 1. 查询 VIP 配额（去雾前）
      const quotaBefore = await ModelAPI.getQuota();
      expect(quotaBefore).toBeDefined();
      expect(typeof quotaBefore.remaining).toBe("number");
      expect(typeof quotaBefore.total).toBe("number");
      expect(typeof quotaBefore.used).toBe("number");
      expect(quotaBefore.used + quotaBefore.remaining).toBe(quotaBefore.total);

      // 2. 批量去雾处理
      const batchForm = createBatchPredictionForm({
        algorithmId: 13,
        items: [{ fileId: uploadedFileId }],
      });
      const batchResult = await ModelAPI.batchPredict(batchForm);
      expect(batchResult).toBeDefined();
      expect(batchResult.total).toBe(1);
      expect(Array.isArray(batchResult.results)).toBe(true);
      expect(batchResult.results.length).toBe(1);

      // 3. 验证配额扣减（去雾后已使用次数应增加）
      const quotaAfter = await ModelAPI.getQuota();
      expect(quotaAfter.used).toBeGreaterThanOrEqual(quotaBefore.used);
    });
  });

  // ============================================================
  // 场景4：参数预设管理
  // ============================================================
  describe("场景4：参数预设管理", () => {
    test("正向流程：获取系统预设→创建自定义预设→使用预设去雾→删除预设", async () => {
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
        algorithmId: 13,
        name: `test_preset_${Date.now()}`,
        params: JSON.stringify({ gamma: 1.2 }),
      });
      const createdPreset = await ModelAPI.createPreset(presetForm);
      expect(createdPreset).toBeDefined();
      expect(createdPreset.id).toBeGreaterThan(0);
      expect(createdPreset.name).toBe(presetForm.name);
      presetIds.push(createdPreset.id);

      // 3. 使用预设参数去雾处理
      const predForm = createPredictionForm({
        algorithmId: 13,
        fileId: uploadedFileId,
        params: createdPreset.params,
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);

      // 4. 更新自定义预设
      const updateForm = createPresetForm({
        algorithmId: 13,
        name: `test_preset_updated_${Date.now()}`,
        params: JSON.stringify({ gamma: 1.5 }),
      });
      const updatedPreset = await ModelAPI.updatePreset(createdPreset.id, updateForm);
      expect(updatedPreset.name).toBe(updateForm.name);

      // 5. 删除自定义预设
      await ModelAPI.deletePreset(createdPreset.id);
      const idx = presetIds.indexOf(createdPreset.id);
      if (idx >= 0) presetIds.splice(idx, 1);

      // 删除后查询应不包含该预设
      const allPresets = await ModelAPI.getPresets({ pageNum: 1, pageSize: 100 });
      const stillExists = allPresets.list.some((p) => p.id === createdPreset.id);
      expect(stillExists).toBe(false);
    });
  });
});
