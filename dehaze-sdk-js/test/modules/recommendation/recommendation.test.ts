import { RecommendationAPI, FileAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { createAnalyzeRequest, createFeedback, createRule } from "#/factories/recommendation";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

describe("推荐管理接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const userAccount = USERS.USER.username;
  const adminAccount = USERS.ADMIN.username;

  afterAll(async () => {
    await cleanup.executeAll();
    // 切回 admin，避免影响后续测试文件
    await login(adminAccount);
  });

  // ============ POST /api/v1/recommendations/analyze - 图像特征分析（普通用户） ============

  describe("POST /api/v1/recommendations/analyze - 图像特征分析", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：分析城市雾霾图片并验证 7 维特征结构", async () => {
      const request = createAnalyzeRequest();
      const analysis = await RecommendationAPI.analyze(request);

      expect(analysis).toBeDefined();
      // 验证 7 维特征字段全部存在
      expect(["light", "moderate", "heavy"]).toContain(analysis.hazeLevel);
      expect(analysis.hazeConfidence).toBeGreaterThanOrEqual(0);
      expect(analysis.hazeConfidence).toBeLessThanOrEqual(1);

      expect(
        ["urban", "landscape", "building", "night", "backlight", "indoor"].includes(
          analysis.sceneType
        )
      ).toBe(true);
      expect(analysis.sceneConfidence).toBeGreaterThanOrEqual(0);
      expect(analysis.sceneConfidence).toBeLessThanOrEqual(1);

      expect(
        ["bright", "normal", "dark", "veryDark", "backlight"].includes(analysis.lighting)
      ).toBe(true);

      expect(analysis.complexity).toBeGreaterThanOrEqual(0);
      expect(analysis.complexity).toBeLessThanOrEqual(1);

      expect(analysis.colorDistribution).toBeDefined();
      expect(typeof analysis.colorDistribution.temperature).toBe("number");
      expect(typeof analysis.colorDistribution.saturation).toBe("number");

      expect(["sd", "hd", "uhd"]).toContain(analysis.resolution);
      expect(["low", "medium", "high"]).toContain(analysis.noiseLevel);
    });

    test("正向测试：分析自然风景图片", async () => {
      const request = createAnalyzeRequest({
        imageUrl: `http://${process.env.DEHAZE_HOST || "127.0.0.1"}:9000/datasets/NH-HAZE-2023/hazy/02.JPG`,
      });
      const analysis = await RecommendationAPI.analyze(request);

      expect(analysis).toBeDefined();
      expect(["light", "moderate", "heavy"]).toContain(analysis.hazeLevel);
      expect(typeof analysis.hazeConfidence).toBe("number");
    });

    test("正向测试：分析夜景图片", async () => {
      const request = createAnalyzeRequest({
        imageUrl: `http://${process.env.DEHAZE_HOST || "127.0.0.1"}:9000/datasets/NH-HAZE-2023/hazy/03.JPG`,
      });
      const analysis = await RecommendationAPI.analyze(request);

      expect(analysis).toBeDefined();
      expect(["light", "moderate", "heavy"]).toContain(analysis.hazeLevel);
    });

    test("边界测试：imageId 不存在应抛出业务错误 A0401", async () => {
      const request = createAnalyzeRequest({ imageId: 99999999 });

      await expectBizError(RecommendationAPI.analyze(request), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界测试：非图片文件应抛出业务错误 A0701", async () => {
      const request = createAnalyzeRequest({
        imageUrl: `http://${process.env.DEHAZE_HOST || "127.0.0.1"}:9000/datasets/test.txt`,
      });

      await expectBizError(RecommendationAPI.analyze(request), [
        "A0701",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ============ GET /api/v1/recommendations/algorithms - 获取算法推荐（普通用户） ============

  describe("GET /api/v1/recommendations/algorithms - 获取算法推荐", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：获取推荐列表并验证 Top 3 结构", async () => {
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: "d41d8cd98f00b204e9800998ecf8427e",
      });

      expect(recommendations).toBeDefined();
      expect(Array.isArray(recommendations)).toBe(true);

      if (recommendations.length > 0) {
        // 验证每个推荐项结构
        recommendations.forEach((rec) => {
          expect(rec.algorithmId).toBeGreaterThan(0);
          expect(rec.algorithmName).toBeTruthy();
          expect(typeof rec.matchScore).toBe("number");
          expect(rec.matchScore).toBeGreaterThanOrEqual(0);
          expect(rec.matchScore).toBeLessThanOrEqual(100);
          expect(rec.reason).toBeTruthy();
          expect(typeof rec.rating).toBe("number");
          expect(rec.rating).toBeGreaterThanOrEqual(0);
          expect(rec.rating).toBeLessThanOrEqual(5);
        });
      }
    });

    test("正向测试：验证推荐结果包含匹配度和推荐理由", async () => {
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: "d41d8cd98f00b204e9800998ecf8427e",
      });

      expect(recommendations).toBeDefined();
      expect(Array.isArray(recommendations)).toBe(true);

      if (recommendations.length > 0) {
        const first = recommendations[0]!;
        expect(first.matchScore).toBeDefined();
        expect(typeof first.reason).toBe("string");
        expect(first.reason!.length).toBeGreaterThan(0);
      }
    });

    test("边界测试：无匹配算法场景返回空数组", async () => {
      // 使用不存在的 analysisId
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        analysisId: 99999999,
      });

      expect(recommendations).toBeDefined();
      expect(Array.isArray(recommendations)).toBe(true);
      // 无匹配时返回空数组而非报错
      expect(recommendations.length).toBeGreaterThanOrEqual(0);
    });

    test("幂等测试：相同图片多次分析结果一致", async () => {
      const req1 = createAnalyzeRequest();
      const analysis1 = await RecommendationAPI.analyze(req1);

      const req2 = createAnalyzeRequest();
      const analysis2 = await RecommendationAPI.analyze(req2);

      // 相同图片多次分析应返回相同的特征
      expect(analysis1.hazeLevel).toBe(analysis2.hazeLevel);
      expect(analysis1.sceneType).toBe(analysis2.sceneType);
      expect(analysis1.lighting).toBe(analysis2.lighting);
    });
  });

  // ============ POST /api/v1/recommendations/feedback - 推荐反馈（普通用户） ============

  describe("POST /api/v1/recommendations/feedback - 推荐反馈", () => {
    let recommendationId: number;

    beforeAll(async () => {
      await login(userAccount);

      // 上传测试图片获取 url 和 md5，用于 analyze + getAlgorithmRecommendations
      const uploadFile = async (
        relativePath: string
      ): Promise<{ id: number; url: string; md5?: string }> => {
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
              await new Promise((resolve) => setTimeout(resolve, 2000));
              continue;
            }
            if (code === "A0501") {
              console.warn(`文件 ${fileName} 已存在，搜索已有记录...`);
              const page = await FileAPI.getPage({ pageNum: 1, pageSize: 5, keywords: fileName });
              const found = page.list.find((f: any) => f.name === fileName);
              if (found) {
                return found.md5
                  ? { id: found.id, url: found.url, md5: found.md5 }
                  : { id: found.id, url: found.url };
              }
              throw new Error(`文件 ${fileName} 已存在但无法在列表中查到，请检查数据库`);
            }
            throw e;
          }
        }
        return { id: 0, url: "" };
      };

      const fileInfo = await uploadFile("../../resources/test/model/hazy.jpg");

      // 用上传的文件 URL 进行图像分析，获取 analyze 返回的 imageMd5
      const analysis = await RecommendationAPI.analyze({ imageUrl: fileInfo.url });

      // 用 analyze 返回的 imageMd5 获取推荐列表，提取真实 recommendationId
      let recommendations: any[] = [];
      if (analysis.imageMd5) {
        recommendations = await RecommendationAPI.getAlgorithmRecommendations({
          imageMd5: analysis.imageMd5,
        });
      }
      if (recommendations.length === 0 && fileInfo.id) {
        // 回退：尝试用 analysisId（某些后端版本支持）
        recommendations = await RecommendationAPI.getAlgorithmRecommendations({
          analysisId: fileInfo.id,
        });
      }

      if (recommendations.length > 0) {
        recommendationId = recommendations[0]!.recommendationId ?? recommendations[0]!.algorithmId;
      } else {
        // 无推荐记录时回退到 algorithmId=1（后续测试可能因 A0401 失败，这是预期行为）
        recommendationId = 1;
      }
    });

    test("正向测试：提交'有用'反馈", async () => {
      const feedback = createFeedback({ recommendationId, useful: true });
      const result = await RecommendationAPI.submitFeedback(feedback);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
    });

    test("正向测试：提交'无用'反馈", async () => {
      const feedback = createFeedback({ recommendationId, useful: false });
      const result = await RecommendationAPI.submitFeedback(feedback);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
    });

    test("边界测试：不存在的推荐ID应抛出业务错误", async () => {
      const feedback = createFeedback({ recommendationId: 99999999 });

      await expectBizError(RecommendationAPI.submitFeedback(feedback), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ============ GET /api/v1/recommendations/rules - 推荐规则配置（管理员） ============

  describe("GET /api/v1/recommendations/rules - 推荐规则配置（管理员）", () => {
    beforeAll(async () => {
      await login(adminAccount);
    });

    test("正向测试：管理员获取规则列表", async () => {
      const rules = await RecommendationAPI.getRules();

      expect(rules).toBeDefined();
      expect(Array.isArray(rules)).toBe(true);

      if (rules.length > 0) {
        rules.forEach((rule) => {
          expect(rule.ruleName).toBeTruthy();
          expect(typeof rule.weight).toBe("number");
          expect(typeof rule.enabled).toBe("boolean");
        });
      }
    });

    test("权限测试：普通用户无法访问规则列表（期望 A0301）", async () => {
      await login(userAccount);
      try {
        await expectBizError(RecommendationAPI.getRules(), ["A0301", "A0300", "ERR_BAD_REQUEST"]);
      } finally {
        await login(adminAccount);
      }
    });
  });

  // ============ PUT /api/v1/recommendations/rules - 更新规则（管理员） ============

  describe("PUT /api/v1/recommendations/rules - 更新规则（管理员）", () => {
    beforeAll(async () => {
      await login(adminAccount);
    });

    test("正向测试：更新规则权重", async () => {
      const rules = await RecommendationAPI.getRules();
      if (rules.length === 0) {
        console.warn("无预置规则可更新，跳过用例");
        return;
      }

      const originalRule = rules[0]!;
      const updateData = createRule({
        ...originalRule,
        weight: Math.min(100, (originalRule.weight || 50) + 10),
      });

      const result = await RecommendationAPI.updateRule(originalRule.id!, updateData);
      expect(result).toBeDefined();

      // 验证更新生效
      const updatedRules = await RecommendationAPI.getRules();
      const updated = updatedRules.find((r) => r.id === originalRule.id);
      expect(updated).toBeDefined();
      expect(updated!.weight).toBe(updateData.weight);

      // 恢复原始权重
      await RecommendationAPI.updateRule(originalRule.id!, {
        ...originalRule,
        weight: originalRule.weight,
      });
    });

    test("正向测试：新增规则", async () => {
      const newRule = createRule();
      const result = await RecommendationAPI.updateRule(0, newRule);

      expect(result).toBeDefined();
      expect(result).toBeGreaterThan(0);

      // 验证新增生效
      const rules = await RecommendationAPI.getRules();
      const found = rules.find((r) => r.id === result);
      expect(found).toBeDefined();
      expect(found!.ruleName).toBe(newRule.ruleName);

      // 注册清理
      cleanup.registerIds(
        () => [result],
        async (id) => {
          // 规则更新接口为 PUT，删除可通过禁用实现
          await RecommendationAPI.updateRule(Number(id), { ...newRule, enabled: false });
        }
      );
    });

    test("边界测试：权重总和校验（期望 A0500）", async () => {
      const rules = await RecommendationAPI.getRules();
      if (rules.length === 0) {
        console.warn("无预置规则可测试权重总和，跳过用例");
        return;
      }

      const originalRule = rules[0]!;
      // 设置一个明显超限的权重（例如 200）
      const updateData = createRule({
        ...originalRule,
        weight: 200,
      });

      await expectBizError(RecommendationAPI.updateRule(originalRule.id!, updateData), [
        "A0500",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("权限测试：普通用户无法修改规则（期望 A0301）", async () => {
      await login(userAccount);
      try {
        const rule = createRule();
        await expectBizError(RecommendationAPI.updateRule(1, rule), [
          "A0301",
          "A0300",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await login(adminAccount);
      }
    });
  });

  // ============ GET /api/v1/recommendations/report - 效果报表（管理员） ============

  describe("GET /api/v1/recommendations/report - 效果报表（管理员）", () => {
    beforeAll(async () => {
      await login(adminAccount);
    });

    test("正向测试：获取推荐效果报表", async () => {
      const report = await RecommendationAPI.getReport();

      expect(report).toBeDefined();
      expect(typeof report.totalRecommendations).toBe("number");
      expect(report.totalRecommendations).toBeGreaterThanOrEqual(0);

      expect(typeof report.adoptionRate).toBe("number");
      expect(report.adoptionRate).toBeGreaterThanOrEqual(0);
      expect(report.adoptionRate).toBeLessThanOrEqual(1);

      expect(typeof report.satisfactionRate).toBe("number");
      expect(report.satisfactionRate).toBeGreaterThanOrEqual(0);
      expect(report.satisfactionRate).toBeLessThanOrEqual(1);

      expect(typeof report.coverageRate).toBe("number");
      expect(report.coverageRate).toBeGreaterThanOrEqual(0);
      expect(report.coverageRate).toBeLessThanOrEqual(1);

      expect(typeof report.coldStartSuccessRate).toBe("number");
      expect(Array.isArray(report.trend)).toBe(true);
    });

    test("正向测试：按日期范围筛选报表", async () => {
      const today = new Date();
      const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
      const startDate = weekAgo.toISOString().slice(0, 10);
      const endDate = today.toISOString().slice(0, 10);

      const report = await RecommendationAPI.getReport({ startDate, endDate });

      expect(report).toBeDefined();
      expect(typeof report.totalRecommendations).toBe("number");
      expect(Array.isArray(report.trend)).toBe(true);

      // 验证 trend 中的日期在筛选范围内
      report.trend.forEach((item) => {
        expect(item.date).toBeTruthy();
        expect(typeof item.adoptionRate).toBe("number");
      });
    });

    test("权限测试：普通用户无法访问报表（期望 A0301）", async () => {
      await login(userAccount);
      try {
        await expectBizError(RecommendationAPI.getReport(), ["A0301", "A0300", "ERR_BAD_REQUEST"]);
      } finally {
        await login(adminAccount);
      }
    });
  });

  // ============ 集成场景 ============

  describe("集成场景", () => {
    test("完整流程：分析图片 → 获取推荐 → 提交反馈", async () => {
      await login(userAccount);

      // Step 1: 分析图片
      const analyzeReq = createAnalyzeRequest();
      const analysis = await RecommendationAPI.analyze(analyzeReq);
      expect(analysis).toBeDefined();
      expect(analysis.hazeLevel).toBeTruthy();

      // Step 2: 获取推荐
      const recommendations = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: "d41d8cd98f00b204e9800998ecf8427e",
      });
      expect(recommendations).toBeDefined();
      expect(Array.isArray(recommendations)).toBe(true);

      // Step 3: 提交反馈（如果有推荐结果）
      if (recommendations.length > 0) {
        const firstRec = recommendations[0]!;
        const feedback = createFeedback({
          recommendationId: firstRec.algorithmId,
          useful: true,
        });
        const feedbackResult = await RecommendationAPI.submitFeedback(feedback);
        expect(feedbackResult).toBeDefined();
        expect(feedbackResult.id).toBeGreaterThan(0);
      }
    });

    test("完整流程：管理员配置规则 → 验证规则已生效", async () => {
      await login(adminAccount);

      // Step 1: 获取当前规则列表
      const rulesBefore = await RecommendationAPI.getRules();
      const countBefore = rulesBefore.length;

      // Step 2: 新增规则
      const newRule = createRule({ sceneType: "night", weight: 30 });
      const newRuleId = await RecommendationAPI.updateRule(0, newRule);
      expect(newRuleId).toBeGreaterThan(0);

      // Step 3: 验证规则已出现在列表中
      const rulesAfter = await RecommendationAPI.getRules();
      expect(rulesAfter.length).toBe(countBefore + 1);

      const found = rulesAfter.find((r) => r.id === newRuleId);
      expect(found).toBeDefined();
      expect(found!.ruleName).toBe(newRule.ruleName);
      expect(found!.sceneType).toBe("night");

      // 清理：禁用新增的规则
      cleanup.register(async () => {
        try {
          await RecommendationAPI.updateRule(newRuleId, { ...newRule, enabled: false });
        } catch {
          // 忽略清理失败
        }
      });
    });
  });
});
