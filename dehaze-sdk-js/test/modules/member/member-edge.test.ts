import { MemberAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createGrowthAdjustForm,
  createLevelAdjustForm,
  createMemberQuery,
} from "#/factories/member";
import { USERS } from "#/factories/constants";

/**
 * 会员管理模块边界强化测试（对抗性脏语料 / 参数边界 / 越权 / 并发 / 性能烟测）。
 *
 * 与 member.test.ts 互补：本文件聚焦任务书指定的强化场景，
 * 管理端写操作统一以 VIP1（id=6）为目标并在 afterAll 恢复其成长值。
 */
describe("会员管理模块边界强化测试", () => {
  const targetUser = USERS.VIP1;

  afterAll(async () => {
    // 恢复 VIP1 预置成长值，避免影响 member.test.ts 与后续文件
    try {
      await login(USERS.ADMIN.username);
      const detail = await MemberAPI.getDetail(targetUser.id);
      const delta = targetUser.member!.growthValue - detail.growthValue;
      if (delta !== 0) {
        await MemberAPI.adjustGrowth(targetUser.id, createGrowthAdjustForm({ changeValue: delta }));
      }
    } catch (e) {
      console.warn("恢复 VIP1 成长值失败:", e);
    }
  });

  // ============ 越权：普通用户调用管理接口 ============

  describe("越权 - 普通用户操作他人会员数据应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    test("边界：普通用户修改他人成长值应失败", async () => {
      await expectBizError(
        MemberAPI.adjustGrowth(targetUser.id, createGrowthAdjustForm({ changeValue: 10 })),
        ["A0301"]
      );
    });

    test("边界：普通用户调整他人等级应失败", async () => {
      await expectBizError(
        MemberAPI.adjustLevel(targetUser.id, createLevelAdjustForm({ levelCode: "level_1" })),
        ["A0301"]
      );
    });
  });

  // ============ 参数边界 ============

  describe("参数边界校验", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：成长值变动值为 0 应拒绝（无意义流水不应落库）", async () => {
      await expectBizError(
        MemberAPI.adjustGrowth(targetUser.id, createGrowthAdjustForm({ changeValue: 0 })),
        ["A0400"]
      );
    });

    test("边界：签到日历 month=13 应拒绝", async () => {
      await login(targetUser.username);
      await expectBizError(MemberAPI.getSignInCalendar(2026, 13), ["A0400"]);
    });

    test("边界：会员分页 pageSize=101 超上限应拒绝", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(MemberAPI.getPage(createMemberQuery({ pageNum: 1, pageSize: 101 })), [
        "A0400",
      ]);
    });
  });

  // ============ 对抗性脏语料 ============

  describe("对抗性脏语料 - 调整原因与关键字搜索", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：调整原因含 emoji/零宽字符/CRLF/全半角混排，流水原样保留", async () => {
      const dirtyReason =
        "审计🚀测试😀\u200b\u200b全角ＡＢＣ１２３半角abc123\r\n换行Tab\t混排" + "长".repeat(100);
      await MemberAPI.adjustGrowth(
        targetUser.id,
        createGrowthAdjustForm({ changeValue: 5, reason: dirtyReason })
      );

      // 以目标用户身份核对流水：reason 原样、operatorId 为 admin（操作审计）
      await login(targetUser.username);
      const logs = await MemberAPI.getGrowthLogs({
        pageNum: 1,
        pageSize: 10,
        changeType: "admin_adjust",
      });
      const target = logs.list.find((l) => l.reason === dirtyReason);
      expect(target, "脏语料 reason 未在流水中原样保留").toBeDefined();
      expect(target!.operatorId).toBe(USERS.ADMIN.id);
    });

    test("暴露：调整原因超长（300 字符超出 DB varchar(256)）应被参数校验拒绝而非 500", async () => {
      // 已知缺陷暴露用例：后端 reason 无 max_length 校验，超长直接落库会触发
      // MySQL DataError → 500，前端只能拿到传输层错误而非业务错误码信封。
      // 修复方向：MemberGrowthAdjustForm.reason 增加 max_length=256（或落库前截断），
      // 修复后本用例应转为通过。
      const overlong = "超".repeat(300);
      await expectBizError(
        MemberAPI.adjustGrowth(
          targetUser.id,
          createGrowthAdjustForm({ changeValue: 1, reason: overlong })
        ),
        ["A0400"]
      );
    });

    test("边界：关键字含 LIKE 通配符/引号/全角应按字面匹配不膨胀结果集", async () => {
      await login(USERS.ADMIN.username);
      const all = await MemberAPI.getPage(createMemberQuery({ pageNum: 1, pageSize: 1 }));
      expect(all.total).toBeGreaterThan(0);

      for (const keyword of ["%", "_", "%_%", "'", '"', "\\", "％"]) {
        const page = await MemberAPI.getPage(
          createMemberQuery({ pageNum: 1, pageSize: 10, keywords: keyword })
        );
        expect(page.total, `keywords=${keyword} 疑似未转义（命中全量 ${all.total}）`).toBeLessThan(
          all.total
        );
      }
    });
  });

  // ============ 并发签到 ============

  describe("并发 - 同一用户并发签到", () => {
    test("并发：同一用户并发签到仅成功一次（唯一索引/预检兜底，T-MM-076）", async () => {
      await login(USERS.USER.username);
      const now = new Date();
      const calendar = await MemberAPI.getSignInCalendar(now.getFullYear(), now.getMonth() + 1);
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
      const alreadySigned = calendar.signDates.includes(today);

      const results = await Promise.allSettled([MemberAPI.signIn(), MemberAPI.signIn()]);
      const errors: any[] = [];
      for (const r of results) {
        if (r.status === "rejected") {
          errors.push(r.reason?.response?.data ?? r.reason);
        }
      }

      if (alreadySigned) {
        // 当日已有签到记录（重跑）：并发请求必须全部被 A0512 拒绝
        expect(errors.length).toBe(2);
        for (const e of errors) {
          expect(e.code).toBe("A0512");
        }
      } else {
        // 当日首次：至多一次成功，失败方必须返回 A0512（预检或唯一约束捕获）
        expect(results.filter((r) => r.status === "fulfilled").length).toBeLessThanOrEqual(1);
        for (const e of errors) {
          expect(e.code).toBe("A0512");
        }
      }
    });
  });

  // ============ 性能烟测 ============

  describe("性能烟测", () => {
    test("会员分页列表响应 <500ms（T-MM-002）", async () => {
      await login(USERS.ADMIN.username);
      const start = Date.now();
      await MemberAPI.getPage(createMemberQuery({ pageNum: 1, pageSize: 10 }));
      expect(Date.now() - start).toBeLessThan(500);
    });

    test("权益概览聚合响应 <200ms（T-MM-097 性能口径）", async () => {
      await login(targetUser.username);
      const start = Date.now();
      await MemberAPI.getBenefitSummary();
      expect(Date.now() - start).toBeLessThan(200);
    });
  });

  // ============ 会员详情越权收紧与管理端详情弹窗数据接口 ============

  describe("会员详情越权收紧 - GET /members/{userId}", () => {
    test("边界：普通用户查询他人详情应被拒（A0301）", async () => {
      await login(USERS.USER.username);
      await expectBizError(MemberAPI.getDetail(targetUser.id), ["A0301"]);
    });

    test("正向：普通用户查询本人详情可见", async () => {
      await login(USERS.USER.username);
      const detail = await MemberAPI.getDetail(USERS.USER.id);
      expect(detail.userId).toBe(USERS.USER.id);
    });
  });

  describe("管理端详情弹窗数据接口 - 成长值流水/消费记录/权益使用/操作日志", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向：目标会员成长值流水（admin 视角按 userId 查）", async () => {
      const logs = await MemberAPI.getAdminGrowthLogs(targetUser.id, {
        pageNum: 1,
        pageSize: 10,
      });
      expect(Array.isArray(logs.list)).toBe(true);
      expect(logs.total).toBeGreaterThanOrEqual(0);
      for (const log of logs.list) {
        expect(typeof log.changeType).toBe("string");
        expect(typeof log.balance).toBe("number");
      }
    });

    test("正向：目标会员消费记录返回订单 VO 结构", async () => {
      const records = await MemberAPI.getConsumptionRecords(targetUser.id, {
        pageNum: 1,
        pageSize: 10,
      });
      expect(Array.isArray(records.list)).toBe(true);
      expect(records.total).toBeGreaterThanOrEqual(0);
      for (const record of records.list) {
        expect(record.orderNo).toBeTruthy();
        expect(typeof record.payableAmount).toBe("number");
      }
    });

    test("正向：目标会员权益使用明细与用户端 benefit-summary 同构", async () => {
      const usage = await MemberAPI.getBenefitUsage(targetUser.id);
      expect(usage.imageCategory.details).toHaveLength(7);
      expect(usage.evaluateCategory).toBeDefined();
      expect(usage.aiCategory).toBeDefined();
    });

    test("正向：目标会员操作日志（Mongo 审计，倒序）", async () => {
      // 目标用户历史用例已产生 level_change/growth_change 审计
      const logs = await MemberAPI.getOperationLogs(targetUser.id, {
        pageNum: 1,
        pageSize: 10,
      });
      expect(logs.total).toBeGreaterThanOrEqual(0);
      for (const log of logs.list) {
        expect(typeof log.id).toBe("string");
        expect(typeof log.operatorId).toBe("number");
        expect(typeof log.createTime).toBe("string");
      }
    });

    test("边界：普通用户调用管理端数据接口应被拒（member:list）", async () => {
      await login(USERS.USER.username);
      await expectBizError(
        MemberAPI.getAdminGrowthLogs(targetUser.id, { pageNum: 1, pageSize: 10 }),
        ["A0301"]
      );
      await expectBizError(
        MemberAPI.getConsumptionRecords(targetUser.id, { pageNum: 1, pageSize: 10 }),
        ["A0301"]
      );
      await expectBizError(MemberAPI.getBenefitUsage(targetUser.id), ["A0301"]);
      await expectBizError(
        MemberAPI.getOperationLogs(targetUser.id, { pageNum: 1, pageSize: 10 }),
        ["A0301"]
      );
    });
  });
});
