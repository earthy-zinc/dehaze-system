import { MemberAPI, OrderAPI, PackageAPI } from "../../../index";
import { PackageStatus } from "@/api/package/model";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createOrderCreateForm,
  createOrderQuery,
  createRefundApplyForm,
  createRefundQuery,
} from "#/factories/order";
import { createPackageForm, yuan } from "#/factories/package";
import { createGrowthAdjustForm, createLevelAdjustForm } from "#/factories/member";
import { TestCleanupRegistry, deletePackageOrOffline } from "#/utils/cleanup";
import { ensureBalance } from "#/utils/mysql";
import { USERS } from "#/factories/constants";

/**
 * 订单模块边界与安全强化（对照 测试用例.md §3.2/3.4/3.6/3.7 与 §6 安全性）：
 * 状态机非法流转、越权、支付安全（余额不足/金额快照一致性）、会员履约、
 * 对抗性脏语料、分页边界、性能烟测。
 */

describe("订单模块边界与安全强化", () => {
  const cleanup = new TestCleanupRegistry();
  const createdOrderNos: string[] = [];
  const createdPackageIds: number[] = [];
  const userAccount = USERS.USER.username;

  // 常规会员卡（level_1，余额可支付）
  let vipPackageId: number;
  // 天价套餐：任何账号余额必不足，用于 A053B 严格断言
  let overpricedPackageId: number;
  // 积分卡
  let creditPackageId: number;

  async function createPackageWithCleanup(status: PackageStatus, overrides = {}): Promise<number> {
    const pkgForm = createPackageForm({ status, ...overrides });
    await PackageAPI.add(pkgForm);
    const page = await PackageAPI.getPage({ name: pkgForm.name, pageNum: 1, pageSize: 10 });
    const created = page.list.find((p) => p.name === pkgForm.name);
    if (!created?.id) throw new Error("订单边界测试: 未能创建套餐");
    createdPackageIds.push(created.id);
    return created.id;
  }

  async function createOrder(packageId: number): Promise<string> {
    const result = await OrderAPI.create(createOrderCreateForm(packageId));
    createdOrderNos.push(result.orderNo);
    return result.orderNo;
  }

  // 创建订单并余额支付（支付失败直接抛错，不让用例静默降级）
  async function createPaidOrder(packageId: number): Promise<string> {
    const orderNo = await createOrder(packageId);
    await OrderAPI.pay(orderNo, { payMethod: "balance" });
    return orderNo;
  }

  beforeAll(async () => {
    // 余额支付真实扣减，先自愈 USER 余额（上限远小于天价套餐 999999900，不影响 A053B 边界）
    await ensureBalance(USERS.USER.id, yuan(10000));
    await login(USERS.ADMIN.username);
    vipPackageId = await createPackageWithCleanup(1);
    creditPackageId = await createPackageWithCleanup(1, {
      packageType: "credit",
      levelCode: undefined,
      period: undefined,
      periodDays: undefined,
      creditAmount: 1000,
      originalPrice: yuan(19.9),
      salePrice: yuan(1),
    });
    overpricedPackageId = await createPackageWithCleanup(1, {
      originalPrice: 999999900,
      salePrice: 999999900,
    });
  });

  afterAll(async () => {
    cleanup.register(async () => {
      for (const orderNo of [...createdOrderNos].reverse()) {
        try {
          const detail = await OrderAPI.getDetail(orderNo);
          if (detail.status === "pending") {
            await OrderAPI.cancel(orderNo, "测试清理");
          }
        } catch (e) {
          console.warn(`清理失败:`, e);
        }
      }
    });
    cleanup.registerIds(() => createdPackageIds, deletePackageOrOffline);
    // 会员卡支付触发会员升级与成长值累积，需将 USER 恢复到预置状态
    // （level_0 + 成长值 100），避免影响后续测试对普通用户的断言
    await login(USERS.ADMIN.username);
    await cleanup.executeAll();
    try {
      await MemberAPI.adjustLevel(USERS.USER.id, createLevelAdjustForm({ levelCode: "level_0" }));
      const detail = await MemberAPI.getDetail(USERS.USER.id);
      const growthDelta = USERS.USER.member!.growthValue - detail.growthValue;
      if (growthDelta !== 0) {
        await MemberAPI.adjustGrowth(
          USERS.USER.id,
          createGrowthAdjustForm({ changeValue: growthDelta })
        );
      }
    } catch (e) {
      console.warn(`清理失败:`, e);
    }
  });

  // ============ 状态机：非法流转拒绝 ============

  describe("订单状态机", () => {
    test("已支付订单重复支付应拒绝（A0533）", async () => {
      await login(userAccount);
      const orderNo = await createPaidOrder(vipPackageId);

      await expectBizError(OrderAPI.pay(orderNo, { payMethod: "balance" }), "A0533");
    });

    test("已支付订单不可取消（A0531）", async () => {
      await login(userAccount);
      const orderNo = await createPaidOrder(vipPackageId);

      await expectBizError(OrderAPI.cancel(orderNo, "已支付订单取消"), "A0531");
    });

    test("已取消订单再支付应拒绝（A0531）", async () => {
      await login(userAccount);
      const orderNo = await createOrder(vipPackageId);
      await OrderAPI.cancel(orderNo, "取消后再支付");

      await expectBizError(OrderAPI.pay(orderNo, { payMethod: "balance" }), "A0531");
    });

    test("待支付订单不可申请售后（A0531）", async () => {
      await login(userAccount);
      const orderNo = await createOrder(vipPackageId);

      await expectBizError(OrderAPI.applyRefund(orderNo, createRefundApplyForm()), "A0531");
    });

    test("已取消订单不可申请售后（A0531）", async () => {
      await login(userAccount);
      const orderNo = await createOrder(vipPackageId);
      await OrderAPI.cancel(orderNo, "取消后申请售后");

      await expectBizError(OrderAPI.applyRefund(orderNo, createRefundApplyForm()), "A0531");
    });
  });

  // ============ 越权防护 ============

  describe("越权防护（A0530，不泄露存在性）", () => {
    let adminOrderNo: string;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      adminOrderNo = await createOrder(vipPackageId);
    });

    test("普通用户不可取消他人订单", async () => {
      await login(userAccount);
      await expectBizError(OrderAPI.cancel(adminOrderNo, "越权取消"), "A0530");
    });

    test("普通用户不可支付他人订单", async () => {
      await login(userAccount);
      await expectBizError(OrderAPI.pay(adminOrderNo, { payMethod: "balance" }), "A0530");
    });

    test("普通用户不可对他人订单申请售后", async () => {
      await login(userAccount);
      await expectBizError(OrderAPI.applyRefund(adminOrderNo, createRefundApplyForm()), "A0530");
    });
  });

  // ============ 支付安全 ============

  describe("支付安全", () => {
    test("余额不足应拒绝（A053B）", async () => {
      await login(userAccount);
      const orderNo = await createOrder(overpricedPackageId);

      await expectBizError(OrderAPI.pay(orderNo, { payMethod: "balance" }), "A053B");
    });

    test("建单后套餐改价不影响支付金额（金额快照一致性）", async () => {
      await login(userAccount);
      const orderNo = await createOrder(vipPackageId);
      const before = await OrderAPI.getDetail(orderNo);
      const originalPayable = before.payableAmount;
      expect(originalPayable).toBeGreaterThan(0);

      // admin 将套餐售价翻倍，已建订单金额不应受影响
      await login(USERS.ADMIN.username);
      const pkgPage = await PackageAPI.getPage({ pageNum: 1, pageSize: 100 });
      const pkg = pkgPage.list.find((p) => p.id === vipPackageId)!;
      await PackageAPI.update(vipPackageId, {
        ...pkg,
        originalPrice: (pkg.originalPrice ?? pkg.salePrice!) * 2,
        salePrice: pkg.salePrice! * 2,
      } as any);

      await login(userAccount);
      await OrderAPI.pay(orderNo, { payMethod: "balance" });
      const after = await OrderAPI.getDetail(orderNo);
      expect(after.status).toBe("paid");
      expect(after.paidAmount).toBe(originalPayable);
      expect(after.paidAmount).not.toBe(pkg.salePrice! * 2);
    });
  });

  // ============ 支付履约 ============

  describe("支付成功履约（会员卡升级 / 积分卡完成）", () => {
    test("会员卡支付后会员等级提升且成长值增加", async () => {
      await login(userAccount);
      const profileBefore = await MemberAPI.getProfile();
      const growthBefore = profileBefore.growthValue;
      const orderNo = await createPaidOrder(vipPackageId);

      const detail = await OrderAPI.getDetail(orderNo);
      expect(detail.status).toBe("paid");
      expect(detail.packageLevel).toBeTruthy();

      const profileAfter = await MemberAPI.getProfile();
      expect(profileAfter.growthValue).toBeGreaterThan(growthBefore);
      expect(profileAfter.levelCode).toBe("level_1");
    });

    test("积分卡支付后订单直接完成且不影响会员等级", async () => {
      await login(userAccount);
      const profileBefore = await MemberAPI.getProfile();
      const orderNo = await createPaidOrder(creditPackageId);

      const detail = await OrderAPI.getDetail(orderNo);
      expect(detail.status).toBe("completed");
      expect(detail.packageType).toBe("credit");
      expect(detail.creditAmount).toBe(1000);
      expect(detail.packageExpireTime).toBeFalsy();

      const profileAfter = await MemberAPI.getProfile();
      // 积分卡不激活会员权益：level_source 与会员有效期均不受影响（member_service
      // 对非 vip 订单仅累积成长值），等级仅可能因成长值自动升级（全局规则，只升不降）
      expect(profileAfter.levelSource).toBe(profileBefore.levelSource);
      expect(profileAfter.expireTime).toBe(profileBefore.expireTime);
      expect(profileAfter.growthValue).toBeGreaterThan(profileBefore.growthValue);
      const levelRank: Record<string, number> = {
        level_0: 0,
        level_1: 1,
        level_2: 2,
        level_3: 3,
      };
      expect(levelRank[profileAfter.levelCode]!).toBeGreaterThanOrEqual(
        levelRank[profileBefore.levelCode]!
      );
    });
  });

  // ============ 售后参数校验 ============

  describe("售后参数校验", () => {
    test("非法 reasonType 应拒绝（A0400），不进入售后流程", async () => {
      await login(userAccount);
      const orderNo = await createPaidOrder(vipPackageId);

      await expectBizError(
        OrderAPI.applyRefund(orderNo, { reasonType: "no_want" as any }),
        "A0400"
      );
      const detail = await OrderAPI.getDetail(orderNo);
      expect(detail.status).toBe("paid");
    });
  });

  // ============ 对抗性语料与注入 ============

  describe("对抗性语料", () => {
    test("取消原因含 emoji/零宽/CRLF/全半角应成功且原样保存", async () => {
      await login(userAccount);
      const orderNo = await createOrder(vipPackageId);
      const dirtyReason = "不想要了\U0001F600​\r\n买重复了：ＡＢＣ123";

      await OrderAPI.cancel(orderNo, dirtyReason);
      const detail = await OrderAPI.getDetail(orderNo);
      expect(detail.status).toBe("cancelled");
      expect(detail.cancelReason).toBe(dirtyReason);
    });

    test("后台关键词 LIKE 通配符注入不引发异常", async () => {
      await login(USERS.ADMIN.username);
      const result = await OrderAPI.getPage(createOrderQuery({ keywords: "%_'\\\\--" }));
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("非法状态筛选值不引发异常", async () => {
      await login(userAccount);
      const result = await OrderAPI.listMy({ status: "~~bad~~" as any, pageNum: 1, pageSize: 10 });
      expect(Array.isArray(result.list)).toBe(true);
    });
  });

  // ============ 分页参数边界 ============

  describe("分页参数边界（A0400）", () => {
    test("我的订单 pageSize 超上限 101 应拒绝", async () => {
      await login(userAccount);
      await expectBizError(OrderAPI.listMy({ pageNum: 1, pageSize: 101 }), "A0400");
    });

    test("pageNum=0 应拒绝", async () => {
      await login(userAccount);
      await expectBizError(OrderAPI.listMy({ pageNum: 0, pageSize: 10 }), "A0400");
    });

    test("后台分页 pageSize 超上限应拒绝", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(OrderAPI.getPage(createOrderQuery({ pageSize: 101 })), "A0400");
    });
  });

  // ============ 性能烟测 ============

  describe("性能烟测", () => {
    test("我的订单列表响应 < 500ms", async () => {
      await login(userAccount);
      const start = Date.now();
      await OrderAPI.listMy({ pageNum: 1, pageSize: 10 });
      expect(Date.now() - start).toBeLessThan(500);
    });

    test("后台订单分页响应 < 2s", async () => {
      await login(USERS.ADMIN.username);
      const start = Date.now();
      await OrderAPI.getPage(createOrderQuery({ pageNum: 1, pageSize: 20 }));
      expect(Date.now() - start).toBeLessThan(2000);
    });

    test("订单统计响应 < 2s", async () => {
      await login(USERS.ADMIN.username);
      const start = Date.now();
      await OrderAPI.getStats();
      expect(Date.now() - start).toBeLessThan(2000);
    });
  });

  // ============ 后台列表不变量 ============

  describe("后台列表筛选不变量", () => {
    test("按商品类型筛选结果与类型一致", async () => {
      await login(USERS.ADMIN.username);
      for (const packageType of ["vip", "credit"] as const) {
        const result = await OrderAPI.getPage(
          createOrderQuery({ packageType, pageNum: 1, pageSize: 20 })
        );
        for (const order of result.list) {
          expect(order.packageType).toBe(packageType);
        }
      }
    });

    test("退款列表 reasonType 契约字段合法", async () => {
      await login(USERS.ADMIN.username);
      const result = await OrderAPI.listRefunds(createRefundQuery({ pageNum: 1, pageSize: 20 }));
      for (const refund of result.list) {
        if (refund.reasonType) {
          expect(["after_sale", "force_majeure", "merchant", "other"]).toContain(refund.reasonType);
        }
        expect(refund.refundAmount).toBeGreaterThanOrEqual(0);
      }
    });
  });
});
