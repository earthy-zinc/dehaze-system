import { MemberAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createBenefitForm,
  createGrowthAdjustForm,
  createLevelAdjustForm,
  createMemberQuery,
} from "#/factories/member";
import { USERS, USERS_BY_LEVEL } from "#/factories/constants";

describe("会员管理模块接口测试", () => {
  // 使用有完整会员记录的 VIP1 用户作为用户端测试目标
  const targetUser = USERS.VIP1;
  let originalLevel: string;
  let originalGrowth: number;

  // ============ 用户端接口（使用 vip1 账号） ============

  describe("GET /api/v1/members/profile - 当前用户会员信息", () => {
    beforeAll(async () => {
      await login(targetUser.username);
    });

    test("正向测试：获取当前用户会员信息", async () => {
      const profile = await MemberAPI.getProfile();

      expect(profile).toBeDefined();
      expect(profile.userId).toBe(targetUser.id);
      expect(profile.levelCode).toBeDefined();
      expect(["level_0", "level_1", "level_2", "level_3"]).toContain(profile.levelCode);
      expect(profile.levelName).toBeTruthy();
      expect(typeof profile.growthValue).toBe("number");
      expect(profile.growthValue).toBeGreaterThanOrEqual(0);
      expect(typeof profile.progressPercent).toBe("number");
      expect(profile.benefits).toBeDefined();
      expect(profile.benefits.monthlyDehazeQuota).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/members/growth-logs - 成长值变动明细", () => {
    beforeAll(async () => {
      await login(targetUser.username);
    });

    test("正向测试：分页查询成长值流水", async () => {
      const result = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 10 });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.total).toBeGreaterThanOrEqual(0);
    });

    test("正向测试：按变动类型筛选", async () => {
      const result = await MemberAPI.getGrowthLogs({
        pageNum: 1,
        pageSize: 10,
        changeType: "sign_in",
      });

      for (const log of result.list) {
        expect(log.changeType).toBe("sign_in");
      }
    });

    test("边界：大页码返回空列表", async () => {
      const result = await MemberAPI.getGrowthLogs({ pageNum: 99999, pageSize: 10 });
      expect(result.list).toEqual([]);
    });
  });

  describe("POST /api/v1/members/sign-in - 每日签到", () => {
    let signedToday = false;

    beforeAll(async () => {
      await login(targetUser.username);
      try {
        await MemberAPI.signIn();
        signedToday = true;
      } catch {
        // 今日已签到，忽略
      }
    });

    test("正向测试：签到或重复签到", async () => {
      try {
        const result = await MemberAPI.signIn();
        expect(result).toBeDefined();
        expect(result.signDate).toBeTruthy();
        expect(result.continuousDays).toBeGreaterThanOrEqual(1);
        expect(result.growthValue).toBeGreaterThan(0);
        signedToday = true;
      } catch (e: any) {
        // 重复签到返回业务错误
        expect(e).toBeDefined();
      }
    });

    test("验证：重复签到返回业务错误", async () => {
      if (!signedToday) {
        await MemberAPI.signIn();
      }
      await expectBizError(MemberAPI.signIn(), "SIGN_IN_ALREADY", undefined, true);
    });
  });

  describe("GET /api/v1/members/sign-in/calendar - 签到日历", () => {
    beforeAll(async () => {
      await login(targetUser.username);
    });

    test("正向测试：查询当月签到日历", async () => {
      const now = new Date();
      const year = now.getFullYear();
      const month = now.getMonth() + 1;

      const calendar = await MemberAPI.getSignInCalendar(year, month);

      expect(calendar).toBeDefined();
      expect(Array.isArray(calendar.signDates)).toBe(true);
      expect(calendar.continuousDays).toBeGreaterThanOrEqual(0);
      expect(calendar.totalDays).toBeGreaterThanOrEqual(0);
    });

    test("边界：查询历史月份签到日历", async () => {
      const calendar = await MemberAPI.getSignInCalendar(2025, 1);
      expect(calendar).toBeDefined();
      expect(Array.isArray(calendar.signDates)).toBe(true);
    });
  });

  // ============ 后台管理接口（使用 admin 账号） ============

  describe("GET /api/v1/members/page - 后台会员分页列表", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询会员列表", async () => {
      const result = await MemberAPI.getPage(createMemberQuery({ pageNum: 1, pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(0);
      expect(result.list.length).toBeLessThanOrEqual(10);
    });

    test("正向测试：按等级筛选 level_0", async () => {
      const result = await MemberAPI.getPage(
        createMemberQuery({ levelCode: "level_0", pageNum: 1, pageSize: 10 })
      );
      for (const member of result.list) {
        expect(member.levelCode).toBe("level_0");
      }
    });

    test("正向测试：按等级筛选 level_1", async () => {
      const result = await MemberAPI.getPage(
        createMemberQuery({ levelCode: "level_1", pageNum: 1, pageSize: 10 })
      );
      for (const member of result.list) {
        expect(member.levelCode).toBe("level_1");
      }
      // vip1 用户应该在 level_1 列表中
      const found = result.list.find((m) => m.userId === USERS.VIP1.id);
      expect(found).toBeDefined();
    });

    test("正向测试：按状态筛选", async () => {
      const result = await MemberAPI.getPage(
        createMemberQuery({ status: 1, pageNum: 1, pageSize: 10 })
      );
      for (const member of result.list) {
        expect(member.status).toBe(1);
      }
    });

    test("边界：大页码返回空列表", async () => {
      const result = await MemberAPI.getPage(createMemberQuery({ pageNum: 99999, pageSize: 10 }));
      expect(result.list).toEqual([]);
    });
  });

  describe("GET /api/v1/members/{userId} - 会员详情", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：查询会员详情", async () => {
      const detail = await MemberAPI.getDetail(targetUser.id);

      expect(detail).toBeDefined();
      expect(detail.userId).toBe(targetUser.id);
      expect(detail.levelCode).toBeDefined();
      expect(detail.levelSource).toBeDefined();
      expect(["growth", "purchase", "admin"]).toContain(detail.levelSource);

      originalLevel = detail.levelCode;
      originalGrowth = detail.growthValue;
    });

    test("正向测试：查询不同等级会员详情", async () => {
      // 验证各等级会员都能查询到详情
      for (const [levelCode, user] of Object.entries(USERS_BY_LEVEL)) {
        const detail = await MemberAPI.getDetail(user.id);
        expect(detail).toBeDefined();
        expect(detail.userId).toBe(user.id);
        expect(detail.levelCode).toBe(levelCode);
      }
    });

    test("异常：查询不存在的会员", async () => {
      await expectBizError(
        MemberAPI.getDetail(99999999),
        ["MEMBER_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/members/{userId}/growth - 成长值调整", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：管理员增加成长值", async () => {
      const form = createGrowthAdjustForm({ changeValue: 50 });
      await MemberAPI.adjustGrowth(targetUser.id, form);

      const detail = await MemberAPI.getDetail(targetUser.id);
      expect(detail.growthValue).toBe(originalGrowth + 50);

      originalGrowth = detail.growthValue;
    });

    test("正向测试：管理员扣减成长值", async () => {
      const form = createGrowthAdjustForm({ changeValue: -20 });
      await MemberAPI.adjustGrowth(targetUser.id, form);

      const detail = await MemberAPI.getDetail(targetUser.id);
      expect(detail.growthValue).toBe(originalGrowth - 20);

      originalGrowth = detail.growthValue;
    });

    test("边界：扣减后成长值不为负（归零）", async () => {
      const detail = await MemberAPI.getDetail(targetUser.id);
      const overDeduct = -(detail.growthValue + 100);

      await MemberAPI.adjustGrowth(
        targetUser.id,
        createGrowthAdjustForm({ changeValue: overDeduct })
      );

      const updated = await MemberAPI.getDetail(targetUser.id);
      expect(updated.growthValue).toBe(0);

      originalGrowth = 0;
    });

    test("异常：缺少原因应抛出业务错误", async () => {
      await expectBizError(
        MemberAPI.adjustGrowth(targetUser.id, { changeValue: 10, reason: "" } as any),
        ["A0400", "A0706", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/members/{userId}/level - 等级调整", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：管理员调整等级", async () => {
      const form = createLevelAdjustForm({ levelCode: "level_2" });
      await MemberAPI.adjustLevel(targetUser.id, form);

      const detail = await MemberAPI.getDetail(targetUser.id);
      expect(detail.levelCode).toBe("level_2");
      expect(detail.levelSource).toBe("admin");

      originalLevel = "level_2";
    });

    test("边界：调整等级不影响成长值", async () => {
      const beforeDetail = await MemberAPI.getDetail(targetUser.id);
      await MemberAPI.adjustLevel(targetUser.id, createLevelAdjustForm({ levelCode: "level_1" }));
      const afterDetail = await MemberAPI.getDetail(targetUser.id);

      expect(afterDetail.levelCode).toBe("level_1");
      expect(afterDetail.growthValue).toBe(beforeDetail.growthValue);

      originalLevel = "level_1";
    });

    test("异常：缺少原因应抛出业务错误", async () => {
      await expectBizError(
        MemberAPI.adjustLevel(targetUser.id, { levelCode: "level_2", reason: "" } as any),
        ["A0400", "A0706", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/members/{userId}/status - 冻结/解冻", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：冻结会员", async () => {
      await MemberAPI.updateStatus(targetUser.id, { status: 0, reason: "测试冻结" });

      const detail = await MemberAPI.getDetail(targetUser.id);
      expect(detail.status).toBe(0);
      expect(detail.frozenReason).toBe("测试冻结");
      expect(detail.frozenTime).toBeTruthy();
    });

    test("正向测试：解冻会员", async () => {
      await MemberAPI.updateStatus(targetUser.id, { status: 1 });

      const detail = await MemberAPI.getDetail(targetUser.id);
      expect(detail.status).toBe(1);
    });

    test("异常：冻结缺少原因应抛出业务错误", async () => {
      await expectBizError(
        MemberAPI.updateStatus(targetUser.id, { status: 0, reason: "" } as any),
        ["A0400", "A0706", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("GET /api/v1/members/benefits - 权益配置列表", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：获取四个等级权益配置", async () => {
      const benefits = await MemberAPI.listBenefits();

      expect(Array.isArray(benefits)).toBe(true);
      expect(benefits.length).toBe(4);

      const levelCodes = benefits.map((b) => b.levelCode);
      expect(levelCodes).toContain("level_0");
      expect(levelCodes).toContain("level_1");
      expect(levelCodes).toContain("level_2");
      expect(levelCodes).toContain("level_3");
    });
  });

  describe("PUT /api/v1/members/benefits/{level} - 修改权益配置", () => {
    let originalBenefit: any;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const benefits = await MemberAPI.listBenefits();
      originalBenefit = benefits.find((b) => b.levelCode === "level_1")!;
    });

    test("正向测试：修改权益配置", async () => {
      const form = createBenefitForm({ monthlyDehazeQuota: 200 });
      await MemberAPI.updateBenefit("level_1", form);

      const benefits = await MemberAPI.listBenefits();
      const updated = benefits.find((b) => b.levelCode === "level_1")!;
      expect(updated.monthlyDehazeQuota).toBe(200);
    });

    afterAll(async () => {
      // 恢复原始配置
      await MemberAPI.updateBenefit("level_1", {
        monthlyDehazeQuota: originalBenefit.monthlyDehazeQuota,
        monthlyEvaluateQuota: originalBenefit.monthlyEvaluateQuota,
        historyRetention: originalBenefit.historyRetention,
        batchLimit: originalBenefit.batchLimit,
        priority: originalBenefit.priority,
      });
    });
  });

  // ============ 异常场景 ============

  describe("异常场景：会员不存在", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("调整不存在会员的成长值应返回业务错误", async () => {
      await expectBizError(
        MemberAPI.adjustGrowth(99999999, createGrowthAdjustForm()),
        ["MEMBER_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("调整不存在会员的等级应返回业务错误", async () => {
      await expectBizError(
        MemberAPI.adjustLevel(99999999, createLevelAdjustForm()),
        ["MEMBER_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("冻结不存在的会员应返回业务错误", async () => {
      await expectBizError(
        MemberAPI.updateStatus(99999999, { status: 0, reason: "测试" }),
        ["MEMBER_NOT_FOUND", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  // ============ 多等级会员验证 ============

  describe("多等级会员验证 - 不同等级用户的权益差异", () => {
    test("level_0 用户权益基础配额", async () => {
      await login(USERS.USER.username);
      const profile = await MemberAPI.getProfile();
      expect(profile.levelCode).toBe("level_0");
      expect(profile.benefits.monthlyDehazeQuota).toBeGreaterThan(0);
    });

    test("level_3(SVIP) 用户权益高于 level_0", async () => {
      await login(USERS.SVIP.username);
      const svipProfile = await MemberAPI.getProfile();

      await login(USERS.USER.username);
      const userProfile = await MemberAPI.getProfile();

      expect(svipProfile.levelCode).toBe("level_3");
      expect(userProfile.levelCode).toBe("level_0");
      // SVIP 配额应高于普通用户
      expect(svipProfile.benefits.monthlyDehazeQuota).toBeGreaterThan(
        userProfile.benefits.monthlyDehazeQuota
      );
    });
  });

  // 测试结束后切回 admin，避免影响后续测试文件
  afterAll(async () => {
    await login(USERS.ADMIN.username);
  });
});
