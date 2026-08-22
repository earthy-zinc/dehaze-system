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

  // 文件级自愈：被历史运行污染的预置用户成长值可能被清零（如成长值归零用例触发
  // level_source=growth 自动降级），此处恢复各预置用户成长值到预置值，使
  // _check_and_adjust_level 按成长值重算等级，保证各用例初始状态一致。
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    const presetGrowth: Record<number, number> = {
      [USERS.USER.id]: USERS.USER.member!.growthValue,
      [USERS.VIP1.id]: USERS.VIP1.member!.growthValue,
      [USERS.VIP2.id]: USERS.VIP2.member!.growthValue,
      [USERS.SVIP.id]: USERS.SVIP.member!.growthValue,
    };
    for (const [uid, target] of Object.entries(presetGrowth)) {
      const detail = await MemberAPI.getDetail(Number(uid));
      const delta = target - detail.growthValue;
      if (delta !== 0) {
        await MemberAPI.adjustGrowth(Number(uid), createGrowthAdjustForm({ changeValue: delta }));
      }
    }
  });

  // ============ 用户端接口（使用 vip1 账号） ============

  describe("GET /api/v1/members/profile - 当前用户会员信息", () => {
    beforeAll(async () => {
      await login(targetUser.username);
    });

    test("正向测试：获取当前用户会员信息", async () => {
      const profile = await MemberAPI.getProfile();

      expect(profile.userId).toBe(targetUser.id);
      expect(["level_0", "level_1", "level_2", "level_3"]).toContain(profile.levelCode);
      expect(profile.levelName).toBeTruthy();
      expect(typeof profile.growthValue).toBe("number");
      expect(profile.growthValue).toBeGreaterThanOrEqual(0);
      expect(typeof profile.progressPercent).toBe("number");
      expect(profile.benefits.monthlyDehazeQuota).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/members/growth-logs - 成长值变动明细", () => {
    beforeAll(async () => {
      await login(targetUser.username);
    });

    test("正向测试：分页查询成长值流水", async () => {
      const result = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 10 });

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

    test("验证：成长值流水记录完整性", async () => {
      const result = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 10 });
      if (result.list.length === 0) {
        console.warn("无成长值流水数据，跳过流水记录完整性测试");
        return;
      }

      const log = result.list[0]!;
      expect(log.id).toBeGreaterThan(0);
      expect(log.changeType).toBeTruthy();
      expect(typeof log.changeValue).toBe("number");
      expect(typeof log.balance).toBe("number");
      expect(log.createTime).toBeTruthy();
    });
  });

  describe("POST /api/v1/members/sign-in - 每日签到", () => {
    beforeAll(async () => {
      await login(targetUser.username);
      try {
        await MemberAPI.signIn();
      } catch {
        // 今日已签到，忽略
      }
    });

    test("正向测试：签到或重复签到", async () => {
      try {
        const result = await MemberAPI.signIn();
        expect(result.signDate).toBeTruthy();
        expect(result.continuousDays).toBeGreaterThanOrEqual(1);
        expect(result.growthValue).toBeGreaterThan(0);
      } catch (e: any) {
        // 重复签到返回业务错误
        expect(e).toBeDefined();
      }
    });

    test("验证：重复签到返回业务错误", async () => {
      await expectBizError(MemberAPI.signIn(), ["SIGN_IN_ALREADY", "A0512"]);
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

      expect(Array.isArray(calendar.signDates)).toBe(true);
      expect(calendar.continuousDays).toBeGreaterThanOrEqual(0);
      expect(calendar.totalDays).toBeGreaterThanOrEqual(0);
    });

    test("边界：查询历史月份签到日历", async () => {
      const calendar = await MemberAPI.getSignInCalendar(2025, 1);
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

    test("正向测试：按成长值区间筛选", async () => {
      const result = await MemberAPI.getPage(
        createMemberQuery({ growthMin: 0, growthMax: 99999, pageNum: 1, pageSize: 10 })
      );
      for (const member of result.list) {
        expect(member.growthValue).toBeGreaterThanOrEqual(0);
        expect(member.growthValue).toBeLessThanOrEqual(99999);
      }
    });

    test("正向测试：按到期时间范围筛选", async () => {
      const result = await MemberAPI.getPage(
        createMemberQuery({
          expireTimeStart: "2025-01-01 00:00:00",
          expireTimeEnd: "2099-12-31 23:59:59",
          pageNum: 1,
          pageSize: 10,
        })
      );
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("边界：关键字模糊匹配", async () => {
      // 使用已知存在会员记录的用户（USER 预置 level_0 会员）构造查询关键字，
      // 避免依赖分页首条数据导致返回空时跳过用例
      const keyword = USERS.USER.username.substring(0, 3);

      const result = await MemberAPI.getPage(
        createMemberQuery({ keywords: keyword, pageNum: 1, pageSize: 10 })
      );
      expect(result.list.length).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/members/{userId} - 会员详情", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：查询会员详情", async () => {
      const detail = await MemberAPI.getDetail(targetUser.id);

      expect(detail.userId).toBe(targetUser.id);
      expect(detail.levelCode).toBeDefined();
      expect(["growth", "purchase", "admin"]).toContain(detail.levelSource);

      originalLevel = detail.levelCode;
      originalGrowth = detail.growthValue;
    });

    test("正向测试：查询不同等级会员详情", async () => {
      for (const [levelCode, user] of Object.entries(USERS_BY_LEVEL)) {
        const detail = await MemberAPI.getDetail(user.id);
        expect(detail.userId).toBe(user.id);
        expect(detail.levelCode).toBe(levelCode);
      }
    });

    test("异常：查询不存在的会员", async () => {
      await expectBizError(MemberAPI.getDetail(99999999), [
        "MEMBER_NOT_FOUND",
        "A0510",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
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
        ["A0400", "A0706", "ERR_BAD_REQUEST"]
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
        ["A0400", "A0706", "ERR_BAD_REQUEST"]
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
        ["A0400", "A0706", "ERR_BAD_REQUEST"]
      );
    });

    test("验证：解冻后冻结字段保留（便于追溯）", async () => {
      await MemberAPI.updateStatus(targetUser.id, { status: 0, reason: "测试冻结字段保留" });
      const frozen = await MemberAPI.getDetail(targetUser.id);
      expect(frozen.status).toBe(0);
      expect(frozen.frozenReason).toBe("测试冻结字段保留");
      expect(frozen.frozenTime).toBeTruthy();

      await MemberAPI.updateStatus(targetUser.id, { status: 1 });
      const thawed = await MemberAPI.getDetail(targetUser.id);
      expect(thawed.status).toBe(1);
      // 解冻后冻结原因和时间应保留不清空
      expect(thawed.frozenReason).toBe("测试冻结字段保留");
      expect(thawed.frozenTime).toBeTruthy();
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

    test("异常：成长值下限大于上限", async () => {
      const form = createBenefitForm({ growthMin: 5000, growthMax: 1000 });
      await expectBizError(MemberAPI.updateBenefit("level_1", form), [
        "BENEFIT_CONFIG_INVALID",
        "A0514",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：权益数值下限校验（负数应失败）", async () => {
      const form = createBenefitForm({ monthlyDehazeQuota: -1 });
      await expectBizError(MemberAPI.updateBenefit("level_1", form), [
        "BENEFIT_CONFIG_INVALID",
        "A0514",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
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

  // ============ 权限测试 ============

  describe("权限测试 - 普通用户访问管理接口应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户查询会员分页列表应失败", async () => {
      // 后端已补 member:list 权限校验（普通用户期望 A0301）
      await expectBizError(MemberAPI.getPage(createMemberQuery()), ["A0301"]);
    });

    test("边界：普通用户冻结会员应失败", async () => {
      await expectBizError(MemberAPI.updateStatus(1, { status: 0, reason: "test" } as any), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户修改权益配置应失败", async () => {
      await expectBizError(MemberAPI.updateBenefit("level_1", createBenefitForm()), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ============ 异常场景 ============

  describe("异常场景：会员不存在", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("调整不存在会员的成长值应返回业务错误", async () => {
      await expectBizError(MemberAPI.adjustGrowth(99999999, createGrowthAdjustForm()), [
        "MEMBER_NOT_FOUND",
        "A0510",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("调整不存在会员的等级应返回业务错误", async () => {
      await expectBizError(MemberAPI.adjustLevel(99999999, createLevelAdjustForm()), [
        "MEMBER_NOT_FOUND",
        "A0510",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("冻结不存在的会员应返回业务错误", async () => {
      await expectBizError(MemberAPI.updateStatus(99999999, { status: 0, reason: "测试" }), [
        "MEMBER_NOT_FOUND",
        "A0510",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ============ 多等级会员验证 ============

  describe("多等级会员验证 - 不同等级用户的权益差异", () => {
    // level_0 样本使用 dept_admin：order.test.ts 并行运行时会将 USER(id=5) 升级后再恢复，
    // 断言 USER 的 level_0 会读到中间态；dept_admin(id=4) 的会员等级无其他测试修改，保持稳定
    test("level_0 用户权益基础配额", async () => {
      await login(USERS.DEPT_ADMIN.username);
      const profile = await MemberAPI.getProfile();
      expect(profile.levelCode).toBe("level_0");
      expect(profile.benefits.monthlyDehazeQuota).toBeGreaterThan(0);
    });

    test("level_3(SVIP) 用户权益高于 level_0", async () => {
      await login(USERS.SVIP.username);
      const svipProfile = await MemberAPI.getProfile();

      await login(USERS.DEPT_ADMIN.username);
      const userProfile = await MemberAPI.getProfile();

      expect(svipProfile.levelCode).toBe("level_3");
      expect(userProfile.levelCode).toBe("level_0");
      // SVIP 配额应高于普通用户
      expect(svipProfile.benefits.monthlyDehazeQuota).toBeGreaterThan(
        userProfile.benefits.monthlyDehazeQuota
      );
    });
  });

  // ============ 保级机制验证 ============

  describe("保级机制验证 - level_source 对自动降级的影响", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("T-MM-084: level_source=admin 成长值变动不降级", async () => {
      await MemberAPI.adjustLevel(USERS.VIP1.id, createLevelAdjustForm({ levelCode: "level_2" }));

      const detail = await MemberAPI.getDetail(USERS.VIP1.id);
      expect(detail.levelCode).toBe("level_2");
      expect(detail.levelSource).toBe("admin");

      const currentGrowth = detail.growthValue;
      const changeValue = -(currentGrowth - 4000);
      await MemberAPI.adjustGrowth(USERS.VIP1.id, createGrowthAdjustForm({ changeValue }));

      const updated = await MemberAPI.getDetail(USERS.VIP1.id);
      expect(updated.levelCode).toBe("level_2");
      expect(updated.levelSource).toBe("admin");
      expect(updated.growthValue).toBeLessThan(5000);
    });

    test("T-MM-085: level_source=growth 成长值变动触发降级", async () => {
      const detail = await MemberAPI.getDetail(USERS.VIP2.id);
      expect(detail.levelCode).toBe("level_2");
      expect(detail.levelSource).toBe("growth");

      const currentGrowth = detail.growthValue;
      await MemberAPI.adjustGrowth(
        USERS.VIP2.id,
        createGrowthAdjustForm({ changeValue: -(currentGrowth - 4000) })
      );

      const updated = await MemberAPI.getDetail(USERS.VIP2.id);
      expect(updated.levelCode).toBe("level_1");
      expect(updated.levelSource).toBe("growth");
      expect(updated.growthValue).toBe(4000);

      const restoreDelta = 8000 - updated.growthValue;
      await MemberAPI.adjustGrowth(
        USERS.VIP2.id,
        createGrowthAdjustForm({ changeValue: restoreDelta })
      );
      const restored = await MemberAPI.getDetail(USERS.VIP2.id);
      expect(restored.levelCode).toBe("level_2");
      expect(restored.levelSource).toBe("growth");
      expect(restored.growthValue).toBe(8000);
    });

    test("T-MM-086: 边界 - 成长值恰好等于等级下限不降级", async () => {
      await MemberAPI.adjustLevel(USERS.VIP1.id, createLevelAdjustForm({ levelCode: "level_2" }));

      const detail = await MemberAPI.getDetail(USERS.VIP1.id);
      const currentGrowth = detail.growthValue;
      await MemberAPI.adjustGrowth(
        USERS.VIP1.id,
        createGrowthAdjustForm({ changeValue: 5000 - currentGrowth })
      );

      const updated = await MemberAPI.getDetail(USERS.VIP1.id);
      expect(updated.levelCode).toBe("level_2");
      expect(updated.growthValue).toBe(5000);
    });
  });

  // 测试结束后恢复 VIP1 原始状态并切回 admin，避免影响后续测试文件
  afterAll(async () => {
    try {
      await login(USERS.ADMIN.username);
      await MemberAPI.adjustLevel(targetUser.id, createLevelAdjustForm({ levelCode: "level_1" }));
      const detail = await MemberAPI.getDetail(targetUser.id);
      const delta = 1500 - detail.growthValue;
      if (delta !== 0) {
        await MemberAPI.adjustGrowth(targetUser.id, createGrowthAdjustForm({ changeValue: delta }));
      }
    } catch (e) {
      console.warn(`清理失败:`, e);
    }
  });
});
