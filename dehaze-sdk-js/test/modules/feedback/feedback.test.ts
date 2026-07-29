import { FeedbackAPI, ModelAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createFeedbackForm,
  createFeedbackQuery,
  createFeedbackReplyForm,
  createRatingForm,
  createRatingQuery,
} from "#/factories/feedback";
import { createPredictionForm } from "#/factories/model";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { getRedis } from "#/utils/redis";
import { createCompletedPredLog } from "#/utils/mysql";
import { USERS } from "#/factories/constants";

describe("反馈评价模块接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdFeedbackIds: number[] = [];
  const createdRatingIds: number[] = [];
  const userAccount = USERS.USER.username;
  const userId = USERS.USER.id;

  async function ensurePredictionLog(): Promise<number> {
    await login(userAccount);
    try {
      const form = createPredictionForm({ algorithmId: 13 });
      const result = await ModelAPI.predictAndWait(form, {
        intervalMs: 2000,
        timeoutMs: 30000,
      });
      if (result.logId && result.status === 2) {
        return result.logId;
      }
    } catch {
      // 算法服务不可用时回退到直接创建已完成记录
    }
    return await createCompletedPredLog(userId, 13);
  }

  // 重置反馈每日次数限制（DAILY_FEEDBACK_LIMIT=5，测试创建反馈会超限）
  async function resetFeedbackDailyLimit(): Promise<void> {
    const redis = getRedis();
    const keys = await redis.keys("feedback:daily:*");
    if (keys.length > 0) {
      await redis.del(keys);
    }
  }

  afterAll(async () => {
    // 反馈无独立删除接口，关闭即可
    cleanup.register(async () => {
      for (const id of [...createdFeedbackIds].reverse()) {
        try {
          await FeedbackAPI.closeFeedback(id, { closeReason: "测试清理" });
        } catch {
          // 忽略
        }
      }
    });
    // 评价无独立删除接口，无需主动清理
    // 预测日志为只追加日志表，无删除接口，测试产生的记录可保留
    await cleanup.executeAll();
    // 切回 admin，避免影响后续测试文件
    await login(USERS.ADMIN.username);
  });

  // ============ 评价接口 - 用户端（使用 user 账号） ============

  describe("POST /api/v1/feedback/ratings - 提交评分（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：提交 5 星好评", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId, { rating: 5 });
      const result = await FeedbackAPI.createRating(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdRatingIds.push(result.id);
    });

    test("正向测试：提交 1 星差评", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId, {
        rating: 1,
        tags: ["残留雾气", "色彩失真"],
      });
      const result = await FeedbackAPI.createRating(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdRatingIds.push(result.id);
    });

    test("正向测试：匿名提交评价", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId, { isAnonymous: 1 });
      const result = await FeedbackAPI.createRating(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdRatingIds.push(result.id);
    });

    test("异常：处理记录不存在", async () => {
      const form = createRatingForm(99999999);
      await expectBizError(FeedbackAPI.createRating(form), ["A0546", "A0400", "ERR_BAD_REQUEST"]);
    });

    test("异常：重复评价", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId);
      const result = await FeedbackAPI.createRating(form);
      createdRatingIds.push(result.id);

      await expectBizError(FeedbackAPI.createRating(form), ["A0540", "A0400", "ERR_BAD_REQUEST"]);
    });

    test("异常：评分越界", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId, { rating: 10 });
      await expectBizError(FeedbackAPI.createRating(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("PUT /api/v1/feedback/ratings/{id} - 修改评分（user）", () => {
    let testRatingId: number;
    let testPredLogId: number;

    beforeAll(async () => {
      await login(userAccount);
      testPredLogId = await ensurePredictionLog();
      const result = await FeedbackAPI.createRating(createRatingForm(testPredLogId));
      testRatingId = result.id;
      createdRatingIds.push(testRatingId);
    });

    test("正向测试：修改评价内容与评分", async () => {
      const newForm = createRatingForm(testPredLogId, {
        rating: 3,
        comment: "修改后的评价内容",
      });

      await FeedbackAPI.updateRating(testRatingId, newForm);

      const updated = await FeedbackAPI.getRatingByPrediction(testPredLogId);
      expect(updated).toBeDefined();
      expect(updated?.rating).toBe(3);
      expect(updated?.comment).toBe("修改后的评价内容");
    });

    test("异常：评价不存在", async () => {
      const form = createRatingForm(testPredLogId);
      await expectBizError(FeedbackAPI.updateRating(99999999, form), [
        "A0541",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/feedback/ratings/my - 我的评价列表（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：分页查询我的评价", async () => {
      const result = await FeedbackAPI.listMyRatings({ pageNum: 1, pageSize: 10 });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const r of result.list) {
        expect(r.id).toBeGreaterThan(0);
        expect(r.predLogId).toBeGreaterThan(0);
        expect(r.rating).toBeGreaterThanOrEqual(1);
        expect(r.rating).toBeLessThanOrEqual(5);
        expect([0, 1]).toContain(r.isAnonymous);
        expect(r.createTime).toBeTruthy();
      }
    });
  });

  describe("GET /api/v1/feedback/ratings/by-prediction/{id} - 按处理记录查评价（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：查询处理记录的评价", async () => {
      const predLogId = await ensurePredictionLog();
      const createResult = await FeedbackAPI.createRating(createRatingForm(predLogId));
      createdRatingIds.push(createResult.id);

      const result = await FeedbackAPI.getRatingByPrediction(predLogId);

      expect(result).toBeDefined();
      expect(result?.predLogId).toBe(predLogId);
      expect(result?.rating).toBeGreaterThanOrEqual(1);
    });

    test("正向测试：未评价的处理记录返回 undefined", async () => {
      const predLogId = await ensurePredictionLog();
      const result = await FeedbackAPI.getRatingByPrediction(predLogId);

      expect(result === undefined || result === null).toBe(true);
    });

    test("正向测试：匿名评价不展示用户信息", async () => {
      const predLogId = await ensurePredictionLog();
      const createResult = await FeedbackAPI.createRating(
        createRatingForm(predLogId, { isAnonymous: 1 })
      );
      createdRatingIds.push(createResult.id);

      const result = await FeedbackAPI.getRatingByPrediction(predLogId);

      expect(result).toBeDefined();
      expect(result?.isAnonymous).toBe(1);
      expect(result?.userId == null || result?.userId === undefined).toBe(true);
      expect(result?.username == null || result?.username === undefined).toBe(true);
      expect(result?.userAvatar == null || result?.userAvatar === undefined).toBe(true);
    });

    test("异常：越权查询他人处理记录的评价", async () => {
      const predLogId = await ensurePredictionLog();
      const createResult = await FeedbackAPI.createRating(createRatingForm(predLogId));
      createdRatingIds.push(createResult.id);

      await login(USERS.ADMIN.username);
      await expectBizError(FeedbackAPI.getRatingByPrediction(predLogId), [
        "A0503",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
      await login(userAccount);
    });
  });

  // ============ 评价接口 - 后台管理（使用 admin 账号） ============

  describe("GET /api/v1/feedback/ratings/page - 后台评价分页列表（admin）", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询后台评价列表", async () => {
      const result = await FeedbackAPI.listRatings(createRatingQuery({ pageNum: 1, pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const r of result.list) {
        expect(r.id).toBeGreaterThan(0);
        expect(r.userId).toBeGreaterThan(0);
        expect(r.rating).toBeGreaterThanOrEqual(1);
        expect(r.rating).toBeLessThanOrEqual(5);
        expect([0, 1]).toContain(r.isHidden);
      }
    });

    test("正向测试：按评分范围筛选", async () => {
      const result = await FeedbackAPI.listRatings(
        createRatingQuery({ ratingMin: 4, ratingMax: 5, pageNum: 1, pageSize: 10 })
      );
      for (const r of result.list) {
        expect(r.rating).toBeGreaterThanOrEqual(4);
        expect(r.rating).toBeLessThanOrEqual(5);
      }
    });

    test("正向测试：按有评论筛选", async () => {
      const result = await FeedbackAPI.listRatings(
        createRatingQuery({ hasComment: true, pageNum: 1, pageSize: 10 })
      );
      for (const r of result.list) {
        expect(r.comment).toBeTruthy();
      }
    });
  });

  describe("PUT /api/v1/feedback/ratings/{id}/hide - 隐藏评价（admin）", () => {
    let testRatingId: number;

    beforeAll(async () => {
      // 先以 user 创建评价，再切到 admin 执行隐藏
      await login(userAccount);
      const predLogId = await ensurePredictionLog();
      const result = await FeedbackAPI.createRating(createRatingForm(predLogId));
      testRatingId = result.id;
      createdRatingIds.push(testRatingId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：隐藏评价", async () => {
      await FeedbackAPI.hideRating(testRatingId);

      const page = await FeedbackAPI.listRatings(createRatingQuery({ pageNum: 1, pageSize: 100 }));
      const found = page.list.find((r) => r.id === testRatingId);
      expect(found?.isHidden).toBe(1);
    });

    test("异常：评价不存在", async () => {
      await expectBizError(FeedbackAPI.hideRating(99999999), ["A0541", "A0400", "ERR_BAD_REQUEST"]);
    });
  });

  describe("POST /api/v1/feedback/ratings/{id}/reply - 回复评价（admin）", () => {
    let testRatingId: number;

    beforeAll(async () => {
      await login(userAccount);
      const predLogId = await ensurePredictionLog();
      const result = await FeedbackAPI.createRating(createRatingForm(predLogId));
      testRatingId = result.id;
      createdRatingIds.push(testRatingId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：回复评价", async () => {
      const replyContent = `回复内容_${Date.now()}`;
      await FeedbackAPI.replyRating(testRatingId, replyContent);

      // 切回 user 查看我的评价中是否有回复
      await login(userAccount);
      const myRatings = await FeedbackAPI.listMyRatings({ pageNum: 1, pageSize: 100 });
      const found = myRatings.list.find((r) => r.id === testRatingId);
      expect(found?.adminReply).toBe(replyContent);
      expect(found?.replyTime).toBeTruthy();
      // 切回 admin 以便后续测试
      await login(USERS.ADMIN.username);
    });

    test("异常：评价不存在", async () => {
      await expectBizError(FeedbackAPI.replyRating(99999999, "测试回复"), [
        "A0541",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/feedback/ratings/stats - 评价统计（admin）", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：获取评价统计数据结构", async () => {
      const stats = await FeedbackAPI.getRatingStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalRatings).toBe("number");
      expect(typeof stats.averageRating).toBe("number");
      expect(stats.ratingDistribution).toBeDefined();
      expect(Array.isArray(stats.positiveTagRanking)).toBe(true);
      expect(Array.isArray(stats.negativeTagRanking)).toBe(true);
      expect(Array.isArray(stats.algorithmStats)).toBe(true);
    });

    test("正向测试：按时间范围查询统计", async () => {
      const startTime = "2025-01-01 00:00:00";
      const endTime = "2026-12-31 23:59:59";
      const stats = await FeedbackAPI.getRatingStats(startTime, endTime);

      expect(stats).toBeDefined();
      expect(typeof stats.totalRatings).toBe("number");
    });
  });

  // ============ 反馈接口 - 用户端（使用 user 账号） ============

  describe("POST /api/v1/feedback - 提交反馈（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
    });

    test("正向测试：提交建议反馈", async () => {
      const form = createFeedbackForm({ feedbackType: "suggestion" });
      const result = await FeedbackAPI.createFeedback(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdFeedbackIds.push(result.id);
    });

    test("正向测试：提交 bug 报告", async () => {
      const form = createFeedbackForm({ feedbackType: "bug" });
      const result = await FeedbackAPI.createFeedback(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdFeedbackIds.push(result.id);
    });

    test("正向测试：提交投诉反馈", async () => {
      const form = createFeedbackForm({
        feedbackType: "complaint",
        title: "测试投诉",
        content: "投诉内容详述",
      });
      const result = await FeedbackAPI.createFeedback(form);

      expect(result).toBeDefined();
      expect(result.id).toBeGreaterThan(0);
      createdFeedbackIds.push(result.id);
    });

    test("异常：缺少必填字段 title", async () => {
      const form = createFeedbackForm();
      const { title, ...rest } = form;
      await expectBizError(FeedbackAPI.createFeedback(rest as any), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常：缺少必填字段 content", async () => {
      const form = createFeedbackForm();
      const { content, ...rest } = form;
      await expectBizError(FeedbackAPI.createFeedback(rest as any), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/feedback/my - 我的反馈列表（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：分页查询我的反馈", async () => {
      const result = await FeedbackAPI.listMyFeedback({ pageNum: 1, pageSize: 10 });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const f of result.list) {
        expect(f.id).toBeGreaterThan(0);
        expect(f.feedbackType).toBeDefined();
        expect(f.title).toBeTruthy();
        expect(["pending", "processing", "replied", "closed"]).toContain(f.status);
      }
    });
  });

  describe("GET /api/v1/feedback/{id} - 反馈详情（user）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
    });

    test("正向测试：获取反馈详情", async () => {
      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);

      expect(detail).toBeDefined();
      expect(detail.id).toBe(testFeedbackId);
      expect(detail.feedbackType).toBeDefined();
      expect(detail.title).toBeTruthy();
      expect(detail.content).toBeTruthy();
      expect(["pending", "processing", "replied", "closed"]).toContain(detail.status);
      expect(Array.isArray(detail.replies)).toBe(true);
    });

    test("验证：非管理员反馈详情 contact 隐藏", async () => {
      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);

      expect(detail).toBeDefined();
      expect(detail.contact == null || detail.contact === undefined || detail.contact === "").toBe(
        true
      );

      await login(USERS.ADMIN.username);
      const adminDetail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(adminDetail.contact).toBeTruthy();
      await login(userAccount);
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.getFeedbackDetail(99999999), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/feedback/{id}/supplement - 补充说明（user）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
    });

    test("正向测试：补充反馈内容", async () => {
      await FeedbackAPI.supplementFeedback(testFeedbackId, {
        content: "补充说明内容",
      });

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      // 补充说明可能体现在 replies 或内容中
      expect(detail).toBeDefined();
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.supplementFeedback(99999999, { content: "测试" }), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/feedback/{id}/supplement - 补充说明重新打开状态（user→admin→user）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);

      await login(USERS.ADMIN.username);
      await FeedbackAPI.replyFeedback(testFeedbackId, createFeedbackReplyForm());
    });

    test("验证：补充说明后状态从 replied 变为 processing", async () => {
      await login(userAccount);

      const detailBefore = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detailBefore.status).toBe("replied");

      await FeedbackAPI.supplementFeedback(testFeedbackId, {
        content: "补充说明重新打开",
      });

      const detailAfter = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detailAfter.status).toBe("processing");
    });
  });

  // ============ 反馈接口 - 后台管理（使用 admin 账号） ============

  describe("GET /api/v1/feedback/page - 后台反馈分页列表（admin）", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询后台反馈列表", async () => {
      const result = await FeedbackAPI.listFeedback(
        createFeedbackQuery({ pageNum: 1, pageSize: 10 })
      );

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const f of result.list) {
        expect(f.id).toBeGreaterThan(0);
        expect(f.userId).toBeGreaterThan(0);
        expect(f.username).toBeTruthy();
        expect(["suggestion", "bug", "experience", "complaint"]).toContain(f.feedbackType);
        expect(["pending", "processing", "replied", "closed"]).toContain(f.status);
      }
    });

    test("正向测试：按反馈类型筛选", async () => {
      const result = await FeedbackAPI.listFeedback(
        createFeedbackQuery({ feedbackType: "suggestion", pageNum: 1, pageSize: 10 })
      );
      for (const f of result.list) {
        expect(f.feedbackType).toBe("suggestion");
      }
    });

    test("正向测试：按状态筛选", async () => {
      const result = await FeedbackAPI.listFeedback(
        createFeedbackQuery({ status: "pending", pageNum: 1, pageSize: 10 })
      );
      for (const f of result.list) {
        expect(f.status).toBe("pending");
      }
    });
  });

  describe("PUT /api/v1/feedback/{id}/assign - 分配处理人（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      // user 提交反馈，admin 分配
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分配处理人", async () => {
      await FeedbackAPI.assignFeedback(testFeedbackId, { assigneeId: 2 });

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detail.assigneeId).toBe(2);
      expect(["processing", "replied", "pending"]).toContain(detail.status);
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.assignFeedback(99999999, { assigneeId: 2 }), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/feedback/{id}/reply - 回复反馈（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：回复反馈", async () => {
      const replyForm = createFeedbackReplyForm();
      await FeedbackAPI.replyFeedback(testFeedbackId, replyForm);

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detail.replies.length).toBeGreaterThan(0);
      const lastReply = detail.replies[detail.replies.length - 1]!;
      expect(lastReply.content).toBe(replyForm.content);
      expect(lastReply.replierType).toBe(2);
      expect(detail.status).toBe("replied");
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.replyFeedback(99999999, createFeedbackReplyForm()), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/feedback/{id}/tags - 设置反馈标签（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：设置反馈标签", async () => {
      const tags = ["重要", "紧急", "已复现"];
      await FeedbackAPI.updateFeedbackTags(testFeedbackId, tags);

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detail.tags).toEqual(tags);
    });

    test("正向测试：清空反馈标签", async () => {
      await FeedbackAPI.updateFeedbackTags(testFeedbackId, []);

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detail.tags).toEqual([]);
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.updateFeedbackTags(99999999, ["标签"]), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/feedback/{id}/close - 关闭反馈（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
      await resetFeedbackDailyLimit();
      const result = await FeedbackAPI.createFeedback(createFeedbackForm());
      testFeedbackId = result.id;
      createdFeedbackIds.push(testFeedbackId);
      await login(USERS.ADMIN.username);
    });

    test("正向测试：关闭反馈", async () => {
      await FeedbackAPI.closeFeedback(testFeedbackId, { closeReason: "测试关闭" });

      const detail = await FeedbackAPI.getFeedbackDetail(testFeedbackId);
      expect(detail.status).toBe("closed");
      expect(detail.closeReason).toBe("测试关闭");
    });

    test("异常：重复关闭已关闭反馈", async () => {
      await expectBizError(FeedbackAPI.closeFeedback(testFeedbackId, { closeReason: "再次关闭" }), [
        "A0544",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(FeedbackAPI.closeFeedback(99999999, { closeReason: "测试" }), [
        "A0543",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/feedback/stats - 反馈统计（admin）", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：获取反馈统计数据结构", async () => {
      const stats = await FeedbackAPI.getFeedbackStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalFeedback).toBe("number");
      expect(stats.typeDistribution).toBeDefined();
      expect(stats.statusDistribution).toBeDefined();
      expect(typeof stats.averageResponseTime).toBe("number");
      expect(typeof stats.averageCloseTime).toBe("number");
      expect(Array.isArray(stats.moduleDistribution)).toBe(true);
      expect(Array.isArray(stats.topKeywords)).toBe(true);
    });

    test("正向测试：按时间范围查询统计", async () => {
      const startTime = "2025-01-01 00:00:00";
      const endTime = "2026-12-31 23:59:59";
      const stats = await FeedbackAPI.getFeedbackStats(startTime, endTime);

      expect(stats).toBeDefined();
      expect(typeof stats.totalFeedback).toBe("number");
    });
  });
});
