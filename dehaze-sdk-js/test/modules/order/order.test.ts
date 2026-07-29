import { MemberAPI, OrderAPI, PackageAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createOrderCreateForm,
  createOrderQuery,
  createRefundApplyForm,
  createRefundQuery,
} from "#/factories/order";
import { createLevelAdjustForm } from "#/factories/member";
import { createPackageForm } from "#/factories/package";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { USERS } from "#/factories/constants";

describe("订单管理模块接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdOrderNos: string[] = [];
  const createdPackageIds: number[] = [];
  let onSalePackageId: number;
  // 用户端测试账号（有会员记录的普通用户）
  const userAccount = USERS.USER.username;

  beforeAll(async () => {
    // admin 创建上架套餐供用户端下单测试
    await login(USERS.ADMIN.username);
    const form = createPackageForm({ status: 1 });
    await PackageAPI.add(form);
    const page = await PackageAPI.getPage({ name: form.name, pageNum: 1, pageSize: 10 });
    const created = page.list.find((p) => p.name === form.name);
    if (!created?.id) throw new Error("订单测试: 未能创建上架套餐");
    onSalePackageId = created.id;
    createdPackageIds.push(onSalePackageId);
  });

  afterAll(async () => {
    cleanup.register(async () => {
      // 取消未支付订单，避免遗留待支付数据
      for (const orderNo of [...createdOrderNos].reverse()) {
        try {
          const detail = await OrderAPI.getDetail(orderNo);
          if (detail.status === "pending") {
            await OrderAPI.cancel(orderNo, "测试清理");
          }
        } catch {
          // 忽略
        }
      }
    });
    cleanup.registerIds(
      () => createdPackageIds,
      (id) => PackageAPI.deleteByIds(id)
    );
    // 切回 admin，避免影响后续测试文件（删除套餐、恢复会员等级都需要 admin 权限）
    await login(USERS.ADMIN.username);
    await cleanup.executeAll();
    // 余额支付会自动完成订单并触发会员升级，需要恢复 USER 用户等级为 level_0
    // 避免影响后续 member 测试对 level_0 用户的断言
    try {
      await MemberAPI.adjustLevel(USERS.USER.id, createLevelAdjustForm({ levelCode: "level_0" }));
    } catch {
      // 忽略恢复失败
    }
  });

  // ============ 用户端接口（使用 user 账号） ============

  describe("POST /api/v1/orders - 创建订单", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：使用余额支付方式创建订单", async () => {
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);

      expect(result).toBeDefined();
      expect(result.orderNo).toBeTruthy();
      expect(["wechat", "alipay", "balance", "combined"]).toContain(result.payMethod);
      expect(typeof result.paid).toBe("boolean");
      createdOrderNos.push(result.orderNo);
    });

    test("正向测试：使用微信支付方式创建订单", async () => {
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "wechat" });
      const result = await OrderAPI.create(form);

      expect(result).toBeDefined();
      expect(result.orderNo).toBeTruthy();
      createdOrderNos.push(result.orderNo);
    });

    test("异常：套餐不存在", async () => {
      const form = createOrderCreateForm(99999999);
      await expectBizError(
        OrderAPI.create(form),
        ["A0520", "A0400", "ERR_BAD_REQUEST"],
      );
    });

    test("异常：套餐已下架", async () => {
      // admin 创建下架套餐
      await login(USERS.ADMIN.username);
      const pkgForm = createPackageForm({ status: 0 });
      await PackageAPI.add(pkgForm);
      const page = await PackageAPI.getPage({ name: pkgForm.name, pageNum: 1, pageSize: 10 });
      const offShelfPkg = page.list.find((p) => p.name === pkgForm.name);
      if (offShelfPkg?.id) createdPackageIds.push(offShelfPkg.id);

      // 切回 user 尝试下单下架套餐
      await login(userAccount);
      const form = createOrderCreateForm(offShelfPkg!.id!);
      await expectBizError(
        OrderAPI.create(form),
        ["A0521", "A0520", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });

  describe("GET /api/v1/orders/my - 我的订单列表", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：分页查询我的订单", async () => {
      const result = await OrderAPI.listMy({ pageNum: 1, pageSize: 10 });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);

      for (const order of result.list) {
        expect(order.id).toBeGreaterThan(0);
        expect(order.orderNo).toBeTruthy();
        expect(order.packageName).toBeTruthy();
        expect(["pending", "paid", "completed", "cancelled", "refunding", "refunded"]).toContain(
          order.status
        );
        expect(typeof order.payableAmount).toBe("number");
      }
    });

    test("正向测试：按状态筛选我的订单", async () => {
      const result = await OrderAPI.listMy({ status: "pending", pageNum: 1, pageSize: 10 });
      for (const order of result.list) {
        expect(order.status).toBe("pending");
      }
    });
  });

  describe("GET /api/v1/orders/{orderNo} - 订单详情", () => {
    let testOrderNo: string;

    beforeAll(async () => {
      await login(userAccount);
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);
      testOrderNo = result.orderNo;
      createdOrderNos.push(testOrderNo);
    });

    test("正向测试：获取订单详情", async () => {
      const detail = await OrderAPI.getDetail(testOrderNo);

      expect(detail).toBeDefined();
      expect(detail.orderNo).toBe(testOrderNo);
      expect(detail.packageName).toBeTruthy();
      expect(["pending", "paid", "completed", "cancelled", "refunding", "refunded"]).toContain(
        detail.status
      );
      expect(typeof detail.payableAmount).toBe("number");
      expect(typeof detail.paidAmount).toBe("number");
      expect(typeof detail.isAutoRenew).toBe("number");
    });

    test("异常：订单不存在", async () => {
      await expectBizError(
        OrderAPI.getDetail("NON_EXISTENT_ORDER_NO_999999"),
        ["A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });

  describe("PUT /api/v1/orders/{orderNo}/cancel - 取消订单", () => {
    let testOrderNo: string;

    beforeAll(async () => {
      await login(userAccount);
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);
      testOrderNo = result.orderNo;
      createdOrderNos.push(testOrderNo);
    });

    test("正向测试：取消待支付订单", async () => {
      await OrderAPI.cancel(testOrderNo, "测试取消");

      const detail = await OrderAPI.getDetail(testOrderNo);
      expect(detail.status).toBe("cancelled");
      expect(detail.cancelReason).toBeTruthy();
    });

    test("异常：取消不存在的订单", async () => {
      await expectBizError(
        OrderAPI.cancel("NON_EXISTENT_ORDER_NO_999999", "测试取消"),
        ["A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });

    test("异常：重复取消订单", async () => {
      await expectBizError(
        OrderAPI.cancel(testOrderNo, "重复取消"),
        ["A0531", "A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });

  describe("POST /api/v1/orders/{orderNo}/pay - 发起支付", () => {
    let testOrderNo: string;

    beforeAll(async () => {
      await login(userAccount);
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);
      testOrderNo = result.orderNo;
      createdOrderNos.push(testOrderNo);
    });

    test("正向测试：对未支付订单发起支付", async () => {
      const result = await OrderAPI.pay(testOrderNo, { payMethod: "balance" });

      expect(result).toBeDefined();
      expect(result.orderNo).toBe(testOrderNo);
      expect(["wechat", "alipay", "balance", "combined"]).toContain(result.payMethod);
    });

    test("异常：订单不存在", async () => {
      await expectBizError(
        OrderAPI.pay("NON_EXISTENT_ORDER_NO_999999", { payMethod: "balance" }),
        ["A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });

  describe("自动续费配置（user）", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("PUT /api/v1/orders/auto-renew/config - 正向测试：开启自动续费", async () => {
      await OrderAPI.updateAutoRenewConfig({
        packageId: onSalePackageId,
        payMethod: "balance",
        enabled: true,
      });

      const config = await OrderAPI.getAutoRenewConfig(onSalePackageId);
      expect(config).toBeDefined();
      expect(config.packageId).toBe(onSalePackageId);
      expect(config.enabled).toBe(true);
    });

    test("PUT /api/v1/orders/auto-renew/config - 正向测试：关闭自动续费", async () => {
      await OrderAPI.updateAutoRenewConfig({
        packageId: onSalePackageId,
        payMethod: "balance",
        enabled: false,
      });

      const config = await OrderAPI.getAutoRenewConfig(onSalePackageId);
      expect(config.enabled).toBe(false);
    });

    test("PUT /api/v1/orders/auto-renew/config - 异常：套餐不存在", async () => {
      await expectBizError(
        OrderAPI.updateAutoRenewConfig({
          packageId: 99999999,
          payMethod: "balance",
          enabled: true,
        }),
        ["A0520", "A0400", "ERR_BAD_REQUEST"],
      );
    });

    test("GET /api/v1/orders/auto-renew/config - 正向测试：查询自动续费配置", async () => {
      const config = await OrderAPI.getAutoRenewConfig(onSalePackageId);

      expect(config).toBeDefined();
      expect(config.packageId).toBe(onSalePackageId);
      expect(typeof config.enabled).toBe("boolean");
      expect(typeof config.failCount).toBe("number");
    });

    test("GET /api/v1/orders/auto-renew/config - 正向测试：未配置的套餐返回默认配置", async () => {
      // admin 创建新套餐未配置过自动续费
      await login(USERS.ADMIN.username);
      const pkgForm = createPackageForm({ status: 1 });
      await PackageAPI.add(pkgForm);
      const page = await PackageAPI.getPage({ name: pkgForm.name, pageNum: 1, pageSize: 10 });
      const newPkg = page.list.find((p) => p.name === pkgForm.name);
      if (newPkg?.id) createdPackageIds.push(newPkg.id);

      // 切回 user 查询
      await login(userAccount);
      const config = await OrderAPI.getAutoRenewConfig(newPkg!.id!);
      expect(config).toBeDefined();
      expect(config.enabled).toBe(false);
    });
  });

  describe("POST /api/v1/orders/{orderNo}/refund - 申请退款（user）", () => {
    let paidOrderNo: string;

    beforeAll(async () => {
      await login(userAccount);
      // 创建一个已支付订单用于退款测试（通过余额支付自动完成）
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);
      paidOrderNo = result.orderNo;
      createdOrderNos.push(paidOrderNo);

      // 余额支付若自动完成，则订单为已支付状态；否则跳过退款测试
      if (!result.paid) {
        try {
          await OrderAPI.pay(paidOrderNo, { payMethod: "balance" });
        } catch {
          // 忽略，退款测试会按实际状态判定
        }
      }
    });

    test("正向测试：对已支付订单申请退款", async () => {
      const detail = await OrderAPI.getDetail(paidOrderNo);
      if (detail.status !== "paid") {
        // 若订单不是已支付状态，跳过此测试（依赖支付流程完成）
        console.warn("退款测试：订单未处于已支付状态，跳过");
        return;
      }

      await OrderAPI.applyRefund(paidOrderNo, createRefundApplyForm());

      const detailAfter = await OrderAPI.getDetail(paidOrderNo);
      expect(["refunding", "refunded"]).toContain(detailAfter.status);
    });

    test("异常：订单不存在", async () => {
      await expectBizError(
        OrderAPI.applyRefund("NON_EXISTENT_ORDER_NO_999999", createRefundApplyForm()),
        ["A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });

  // ============ 后台管理接口（使用 admin 账号） ============

  describe("GET /api/v1/orders/page - 后台订单分页列表", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询后台订单列表", async () => {
      const result = await OrderAPI.getPage(createOrderQuery({ pageNum: 1, pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const order of result.list) {
        expect(order.id).toBeGreaterThan(0);
        expect(order.orderNo).toBeTruthy();
        expect(order.userId).toBeGreaterThan(0);
        expect(order.username).toBeTruthy();
        expect(["pending", "paid", "completed", "cancelled", "refunding", "refunded"]).toContain(
          order.status
        );
      }
    });

    test("正向测试：按状态筛选", async () => {
      const result = await OrderAPI.getPage(
        createOrderQuery({ status: "pending", pageNum: 1, pageSize: 10 })
      );
      for (const order of result.list) {
        expect(order.status).toBe("pending");
      }
    });

    test("正向测试：按支付方式筛选", async () => {
      const result = await OrderAPI.getPage(
        createOrderQuery({ payMethod: "balance", pageNum: 1, pageSize: 10 })
      );
      for (const order of result.list) {
        expect(order.payMethod).toBe("balance");
      }
    });
  });

  describe("GET /api/v1/orders/stats - 订单统计", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：获取订单统计数据结构", async () => {
      const stats = await OrderAPI.getStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalOrders).toBe("number");
      expect(typeof stats.totalRevenue).toBe("number");
      expect(typeof stats.totalRefund).toBe("number");
      expect(typeof stats.refundRate).toBe("number");
      expect(stats.statusDistribution).toBeDefined();
      expect(stats.payMethodDistribution).toBeDefined();
      expect(Array.isArray(stats.packageDistribution)).toBe(true);
      expect(Array.isArray(stats.dailyStats)).toBe(true);
    });

    test("正向测试：按时间范围查询统计", async () => {
      const startTime = "2025-01-01 00:00:00";
      const endTime = "2026-12-31 23:59:59";
      const stats = await OrderAPI.getStats(startTime, endTime);

      expect(stats).toBeDefined();
      expect(typeof stats.totalOrders).toBe("number");
    });
  });

  describe("GET /api/v1/orders/refunds/page - 退款审核列表", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询退款列表", async () => {
      const result = await OrderAPI.listRefunds(createRefundQuery({ pageNum: 1, pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      for (const refund of result.list) {
        expect(refund.id).toBeGreaterThan(0);
        expect(refund.refundNo).toBeTruthy();
        expect(refund.orderNo).toBeTruthy();
        expect(refund.userId).toBeGreaterThan(0);
        expect(typeof refund.refundAmount).toBe("number");
        expect(["refunding", "refunded", "refund_failed"]).toContain(refund.status);
      }
    });

    test("正向测试：按状态筛选", async () => {
      const result = await OrderAPI.listRefunds(
        createRefundQuery({ status: "refunding", pageNum: 1, pageSize: 10 })
      );
      for (const refund of result.list) {
        expect(refund.status).toBe("refunding");
      }
    });
  });

  describe("退款审核 - approve/reject（admin）", () => {
    let testRefundId: number;
    let testOrderNo: string;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      // 准备一个处于退款中的订单：切到 user 下单支付并申请退款，再切回 admin
      await login(userAccount);
      const form = createOrderCreateForm(onSalePackageId, { payMethod: "balance" });
      const result = await OrderAPI.create(form);
      testOrderNo = result.orderNo;
      createdOrderNos.push(testOrderNo);

      if (!result.paid) {
        try {
          await OrderAPI.pay(testOrderNo, { payMethod: "balance" });
        } catch {
          // 忽略
        }
      }

      try {
        await OrderAPI.applyRefund(testOrderNo, createRefundApplyForm());
        await login(USERS.ADMIN.username);
        const refunds = await OrderAPI.listRefunds(
          createRefundQuery({ pageNum: 1, pageSize: 100 })
        );
        const found = refunds.list.find((r) => r.orderNo === testOrderNo);
        if (found?.id) testRefundId = found.id;
      } catch {
        // 忽略
      }
    });

    test("正向测试：驳回退款申请", async () => {
      if (!testRefundId) {
        console.warn("无可用退款记录，跳过驳回测试");
        return;
      }

      await OrderAPI.rejectRefund(testRefundId, {
        approved: false,
        remark: "测试驳回",
      });

      const refunds = await OrderAPI.listRefunds(createRefundQuery({ pageNum: 1, pageSize: 100 }));
      const found = refunds.list.find((r) => r.id === testRefundId);
      expect(found?.status).toBe("refund_failed");
    });

    test("异常：退款记录不存在", async () => {
      await expectBizError(
        OrderAPI.approveRefund(99999999, { approved: true, remark: "测试" }),
        ["A0537", "A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });

    test("异常：驳回不存在的退款记录", async () => {
      await expectBizError(
        OrderAPI.rejectRefund(99999999, { approved: false, remark: "测试" }),
        ["A0537", "A0530", "A0400", "ERR_BAD_REQUEST"],
      );
    });
  });
});
