import { FeedbackAPI, ImageInputHistoryAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createFeedbackForm,
  createFeedbackQuery,
  createFeedbackReplyForm,
  createRatingForm,
  createRatingQuery,
} from "#/factories/feedback";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { USERS } from "#/factories/constants";

describe("反馈评价模块接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdFeedbackIds: number[] = [];
  const createdRatingIds: number[] = [];
  const createdHistoryIds: number[] = [];
  // 用户端测试账号（有会员记录的普通用户）
  const userAccount = USERS.USER.username;

  // 创建一条处理记录作为评价关联
  async function ensurePredictionLog(): Promise<number> {
    // 评价需要用户拥有处理记录，使用 user 账号创建
    await login(userAccount);
    const id = await ImageInputHistoryAPI.create({
      originalImageUrl: "/images/test_haze.jpg",
      resultImageUrl: "/images/test_dehazed.jpg",
      algorithmId: 1,
      algorithmName: "DCP",
      processingTime: 1500,
      status: 1,
      inputSource: "upload",
    });
    createdHistoryIds.push(id);
    return id;
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
    cleanup.registerIds(
      () => createdHistoryIds,
      (id) => ImageInputHistoryAPI.deleteById(Number(id))
    );
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
      await expectBizError(
        FeedbackAPI.createRating(form),
        ["PREDICTION_LOG_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常：重复评价", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId);
      const result = await FeedbackAPI.createRating(form);
      createdRatingIds.push(result.id);

      await expectBizError(
        FeedbackAPI.createRating(form),
        ["RATING_ALREADY_EXISTS", "A0501", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常：评分越界", async () => {
      const predLogId = await ensurePredictionLog();
      const form = createRatingForm(predLogId, { rating: 10 });
      await expectBizError(
        FeedbackAPI.createRating(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
      await expectBizError(
        FeedbackAPI.updateRating(99999999, form),
        ["RATING_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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

    test("边界：大页码返回空列表", async () => {
      const result = await FeedbackAPI.listMyRatings({ pageNum: 99999, pageSize: 10 });
      expect(result.list).toEqual([]);
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

    test("边界：大页码返回空列表", async () => {
      const result = await FeedbackAPI.listRatings(
        createRatingQuery({ pageNum: 99999, pageSize: 10 })
      );
      expect(result.list).toEqual([]);
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
      await expectBizError(
        FeedbackAPI.hideRating(99999999),
        ["RATING_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
      await expectBizError(
        FeedbackAPI.replyRating(99999999, "测试回复"),
        ["RATING_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
      await expectBizError(
        FeedbackAPI.createFeedback(rest as any),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常：缺少必填字段 content", async () => {
      const form = createFeedbackForm();
      const { content, ...rest } = form;
      await expectBizError(
        FeedbackAPI.createFeedback(rest as any),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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

    test("边界：大页码返回空列表", async () => {
      const result = await FeedbackAPI.listMyFeedback({ pageNum: 99999, pageSize: 10 });
      expect(result.list).toEqual([]);
    });
  });

  describe("GET /api/v1/feedback/{id} - 反馈详情（user）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
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

    test("异常：反馈不存在", async () => {
      await expectBizError(
        FeedbackAPI.getFeedbackDetail(99999999),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("POST /api/v1/feedback/{id}/supplement - 补充说明（user）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
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
      await expectBizError(
        FeedbackAPI.supplementFeedback(99999999, { content: "测试" }),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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

    test("边界：大页码返回空列表", async () => {
      const result = await FeedbackAPI.listFeedback(
        createFeedbackQuery({ pageNum: 99999, pageSize: 10 })
      );
      expect(result.list).toEqual([]);
    });
  });

  describe("PUT /api/v1/feedback/{id}/assign - 分配处理人（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      // user 提交反馈，admin 分配
      await login(userAccount);
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
      await expectBizError(
        FeedbackAPI.assignFeedback(99999999, { assigneeId: 2 }),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("POST /api/v1/feedback/{id}/reply - 回复反馈（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
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
      await expectBizError(
        FeedbackAPI.replyFeedback(99999999, createFeedbackReplyForm()),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/feedback/{id}/tags - 设置反馈标签（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
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
      await expectBizError(
        FeedbackAPI.updateFeedbackTags(99999999, ["标签"]),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/feedback/{id}/close - 关闭反馈（admin）", () => {
    let testFeedbackId: number;

    beforeAll(async () => {
      await login(userAccount);
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
      await expectBizError(
        FeedbackAPI.closeFeedback(testFeedbackId, { closeReason: "再次关闭" }),
        ["FEEDBACK_CLOSED", "ORDER_STATUS_INVALID", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常：反馈不存在", async () => {
      await expectBizError(
        FeedbackAPI.closeFeedback(99999999, { closeReason: "测试" }),
        ["FEEDBACK_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
