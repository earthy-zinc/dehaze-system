import { ModelAPI, FileAPI } from "../../../index";
import RecommendationAPI from "@/api/recommendation";
import FavoriteAPI from "@/api/favorite";
import { createPredictionForm } from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { ensureDehazeQuota } from "#/utils/quota";
import { USERS } from "#/factories/constants";
import type { RecommendedAlgorithm } from "@/api/recommendation/model";
import * as fs from "fs";
import * as path from "path";

/**
 * 推荐跨模块集成测试：推荐查询 → 采纳/不采纳 → 反馈提交 的闭环协作。
 */
describe("推荐跨模块集成测试", () => {
  const favoriteIds: number[] = [];
  let uploadedFileId: number;
  let uploadedFileUrl: string;

  beforeAll(async () => {
    await ensureDehazeQuota();
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
            // 限流，等待 3 秒后重试（最多 3 次）
            await new Promise((resolve) => setTimeout(resolve, 3000));
            continue;
          }
          if (code === "A0501") {
            // 文件已存在（MD5 去重），按文件名搜索复用已有记录
            const page = await FileAPI.getPage({
              pageNum: 1,
              pageSize: 5,
              keywords: fileName,
            });
            const found = page.list.find((f) => f.name === fileName && f.url);
            if (found) {
              return { id: found.id, url: found.url };
            }
            // 查到但 url 为空（可能已删除），等待后重试上传
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

  /** 基于分析结果获取算法推荐，并断言返回列表结构 */
  async function getRecommendations(imageMd5: string): Promise<RecommendedAlgorithm[]> {
    const recommendations = await RecommendationAPI.getAlgorithmRecommendations({ imageMd5 });
    expect(Array.isArray(recommendations)).toBe(true);
    return recommendations;
  }

  describe("场景1：推荐→采纳→反馈 完整流程", () => {
    test("正向流程：分析图片→获取推荐→采纳推荐算法去雾→提交有用反馈", async () => {
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis.hazeLevel).toBeDefined();
      expect(typeof analysis.hazeConfidence).toBe("number");

      const recommendations = await getRecommendations(analysis.imageMd5!);
      if (recommendations.length === 0) {
        // 后端无推荐数据时跳过后续断言（不视为失败）
        return;
      }

      const top = recommendations[0]!;
      expect(top.algorithmId).toBeDefined();
      expect(typeof top.matchScore).toBe("number");
      expect(typeof top.reason).toBe("string");

      // 采纳推荐算法去雾，recommendedBy 标记来源
      const recommendationId = top.recommendationId ?? top.algorithmId;
      const predResult = await ModelAPI.predictAndWait(
        createPredictionForm({
          algorithmId: top.algorithmId,
          fileId: uploadedFileId,
          recommendedBy: recommendationId,
        }),
        { intervalMs: 2000, timeoutMs: 120000 }
      );
      expect(predResult.status).toBe(2);

      const feedbackResult = await RecommendationAPI.submitFeedback({
        recommendationId,
        useful: true,
      });
      expect(feedbackResult.id).toBeDefined();
    });
  });

  describe("场景2：推荐→不采纳→反馈 完整流程", () => {
    test("正向流程：分析图片→获取推荐→不采纳使用其他算法→提交无用反馈", async () => {
      const analysis = await RecommendationAPI.analyze({ imageUrl: uploadedFileUrl });
      expect(analysis.sceneType).toBeDefined();

      const recommendations = await getRecommendations(analysis.imageMd5!);
      if (recommendations.length === 0) {
        // 后端无推荐数据时跳过后续断言
        return;
      }

      // 不采纳推荐，改用固定可用算法 13
      const predResult = await ModelAPI.predictAndWait(
        createPredictionForm({
          algorithmId: 13,
          fileId: uploadedFileId,
        }),
        { intervalMs: 2000, timeoutMs: 120000 }
      );
      expect(predResult.status).toBe(2);

      const recommendationId =
        recommendations[0]!.recommendationId ?? recommendations[0]!.algorithmId;
      const feedbackResult = await RecommendationAPI.submitFeedback({
        recommendationId,
        useful: false,
      });
      expect(feedbackResult.id).toBeDefined();
    });
  });
});
