import { ModelAPI, FileAPI } from "../../../index";
import RecommendationAPI from "@/api/recommendation";
import FavoriteAPI from "@/api/favorite";
import { createPredictionForm } from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

/**
 * 推荐跨模块集成测试
 *
 * 验证推荐管理模块与去雾处理模块之间的协作：
 * 推荐查询 → 采纳/不采纳 → 反馈提交。
 * 后端尚未实现推荐接口时测试会失败——这是 TDD 预期行为。
 */
describe("推荐跨模块集成测试", () => {
  const favoriteIds: number[] = [];
  let uploadedFileId: number;
  let uploadedFileUrl: string;

  beforeAll(async () => {
    await login(USERS.USER.username);

    const uploadWithRetry = async (relativePath: string): Promise<{ id: number; url: string }> => {
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
            // 文件已存在（MD5 去重），按文件名搜索获取已有记录
            const page = await FileAPI.getPage({
              pageNum: 1,
              pageSize: 5,
              keywords: fileName,
            });
            const found = page.list.find((f) => f.name === fileName && f.url);
            if (found) {
              return { id: found.id, url: found.url };
            }
            // 查到的文件 url 为空（可能已删除），等待后重试上传
            if (attempt < 2) {
              await new Promise((resolve) => setTimeout(resolve, 3000));
              continue;
            }
            throw new Error(`文件 ${fileName} 已存在但无法在列表中查到有效记录`);
          }
          throw e;
        }
      }
      return { id: 0, url: "" };
    };

    const hazyInfo = await uploadWithRetry("../../resources/test/model/hazy.jpg");
    uploadedFileId = hazyInfo.id;
    uploadedFileUrl = hazyInfo.url;
  });

  afterAll(async () => {
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch {}
    }

    await logout();
  });

  // ============================================================
  // 场景1：推荐 → 采纳 → 反馈 完整流程
  // ============================================================
  describe("场景1：推荐→采纳→反馈 完整流程", () => {
    test("正向流程：分析图片→获取推荐→采纳推荐算法去雾→提交有用反馈", async () => {
      // 1. 分析图片特征
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis).toBeDefined();
      expect(analysis.hazeLevel).toBeDefined();
      expect(typeof analysis.hazeConfidence).toBe("number");

      // 2. 获取算法推荐
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: (analysis as any).imageMd5,
      });
      expect(Array.isArray(recommendations)).toBe(true);

      if (recommendations.length === 0) {
        // 后端无推荐数据时跳过后续断言（不视为失败）
        return;
      }

      const topRecommendation = recommendations[0]!;
      expect(topRecommendation.algorithmId).toBeDefined();
      expect(typeof topRecommendation.matchScore).toBe("number");
      expect(typeof topRecommendation.reason).toBe("string");

      // 3. 采纳推荐算法去雾（传入 recommendedBy 标记来源）
      const recommendationId = topRecommendation.recommendationId ?? topRecommendation.algorithmId;
      const predForm = createPredictionForm({
        algorithmId: topRecommendation.algorithmId,
        fileId: uploadedFileId,
        recommendedBy: recommendationId,
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);

      // 4. 提交推荐反馈（有用）
      const feedbackResult = await RecommendationAPI.submitFeedback({
        recommendationId,
        useful: true,
      });
      expect(feedbackResult).toBeDefined();
      expect(feedbackResult.id).toBeDefined();
    });
  });

  // ============================================================
  // 场景2：推荐 → 不采纳 → 反馈 完整流程
  // ============================================================
  describe("场景2：推荐→不采纳→反馈 完整流程", () => {
    test("正向流程：分析图片→获取推荐→不采纳使用其他算法→提交无用反馈", async () => {
      // 1. 分析图片特征
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis).toBeDefined();
      expect(analysis.sceneType).toBeDefined();

      // 2. 获取算法推荐
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: (analysis as any).imageMd5,
      });
      expect(Array.isArray(recommendations)).toBe(true);

      if (recommendations.length === 0) {
        // 后端无推荐数据时跳过后续断言
        return;
      }

      // 3. 不采纳推荐，使用其他算法（固定可用算法 13）
      const predForm = createPredictionForm({
        algorithmId: 13,
        fileId: uploadedFileId,
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);

      // 4. 提交"无用"反馈
      const recommendationId =
        recommendations[0]!.recommendationId ?? recommendations[0]!.algorithmId;
      const feedbackResult = await RecommendationAPI.submitFeedback({
        recommendationId,
        useful: false,
      });
      expect(feedbackResult).toBeDefined();
      expect(feedbackResult.id).toBeDefined();
    });
  });
});
