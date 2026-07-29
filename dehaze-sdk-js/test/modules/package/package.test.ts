import { CouponAPI, PackageAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import {
  createCouponForm,
  createCouponQuery,
  createPackageForm,
  createPackageQuery,
} from "#/factories/package";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { USERS } from "#/factories/constants";

describe("套餐管理模块接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdPackageIds: number[] = [];
  const createdCouponIds: number[] = [];
  // 用户端测试账号（有会员记录的普通用户）
  const userAccount = USERS.USER.username;

  afterAll(async () => {
    // 先注册 package 清理，后注册 coupon 清理；
    // executeAll 按 LIFO 顺序执行，coupon 先于 package 删除（coupon.applicableScope 引用 package）
    cleanup.registerIds(
      () => createdPackageIds,
      (id) => PackageAPI.deleteByIds(id)
    );
    cleanup.registerIds(
      () => createdCouponIds,
      (id) => CouponAPI.deleteByIds(id)
    );
    await cleanup.executeAll();
    // 切回 admin，避免影响后续测试文件
    await login(USERS.ADMIN.username);
  });

  // ============ 用户端接口（使用 user 账号） ============

  describe("GET /api/v1/packages - 用户端在售套餐列表", () => {
    beforeAll(async () => {
      await login(userAccount);
    });

    test("正向测试：获取在售套餐列表", async () => {
      const list = await PackageAPI.listOnSale();

      expect(Array.isArray(list)).toBe(true);
      for (const pkg of list) {
        expect(pkg.id).toBeGreaterThan(0);
        expect(pkg.name).toBeTruthy();
        expect(["level_1", "level_2", "level_3"]).toContain(pkg.levelCode);
        expect(["monthly", "quarterly", "yearly"]).toContain(pkg.period);
        expect(pkg.salePrice).toBeGreaterThanOrEqual(0);
        expect(pkg.benefits).toBeDefined();
      }
    });
  });

  describe("GET /api/v1/packages/{id} - 用户端套餐详情", () => {
    let onSalePackageId: number;

    beforeAll(async () => {
      // 切到 admin 创建上架套餐供用户端测试
      await login(USERS.ADMIN.username);
      const form = createPackageForm({ status: 1 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      if (!created?.id) throw new Error("未能创建上架套餐");
      onSalePackageId = created.id;
      createdPackageIds.push(onSalePackageId);
      // 切回 user 测试用户端接口
      await login(userAccount);
    });

    test("正向测试：获取套餐详情", async () => {
      const detail = await PackageAPI.getDetail(onSalePackageId);

      expect(detail).toBeDefined();
      expect(detail.id).toBe(onSalePackageId);
      expect(detail.name).toBeTruthy();
      expect(detail.benefits).toBeDefined();
    });

    test("异常：套餐不存在", async () => {
      await expectBizError(PackageAPI.getDetail(99999999), ["A0520", "A0400", "ERR_BAD_REQUEST"]);
    });
  });

  describe("GET /api/v1/packages/calculate-price - 价格计算", () => {
    let onSalePackageId: number;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createPackageForm({ status: 1 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      if (!created?.id) throw new Error("未能创建上架套餐");
      onSalePackageId = created.id;
      createdPackageIds.push(onSalePackageId);
      await login(userAccount);
    });

    test("正向测试：不使用优惠券计算价格", async () => {
      const result = await PackageAPI.calculatePrice(onSalePackageId);

      expect(result).toBeDefined();
      expect(typeof result.originalPrice).toBe("number");
      expect(typeof result.discountAmount).toBe("number");
      expect(typeof result.couponAmount).toBe("number");
      expect(typeof result.payableAmount).toBe("number");
      expect(result.payableAmount).toBeGreaterThanOrEqual(0);
      expect(result.originalPrice).toBeGreaterThanOrEqual(result.payableAmount);
    });

    test("异常：套餐不存在", async () => {
      await expectBizError(PackageAPI.calculatePrice(99999999), [
        "A0520",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ============ 后台管理接口（使用 admin 账号） ============

  describe("GET /api/v1/packages/page - 后台套餐分页列表", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：分页查询", async () => {
      const result = await PackageAPI.getPage(createPackageQuery({ pageNum: 1, pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);
    });

    test("正向测试：按等级筛选", async () => {
      const result = await PackageAPI.getPage(
        createPackageQuery({ levelCode: "level_1", pageNum: 1, pageSize: 10 })
      );
      for (const pkg of result.list) {
        expect(pkg.levelCode).toBe("level_1");
      }
    });

    test("正向测试：按状态筛选", async () => {
      const result = await PackageAPI.getPage(
        createPackageQuery({ status: 1, pageNum: 1, pageSize: 10 })
      );
      for (const pkg of result.list) {
        expect(pkg.status).toBe(1);
      }
    });
  });

  describe("POST /api/v1/packages - 新增套餐", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：创建下架状态的套餐", async () => {
      const form = createPackageForm({ status: 0 });
      await PackageAPI.add(form);

      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      expect(created).toBeDefined();
      expect(created?.levelCode).toBe(form.levelCode);
      expect(created?.period).toBe(form.period);
      expect(created?.status).toBe(0);

      if (created?.id) createdPackageIds.push(created.id);
    });

    test("正向测试：创建不同周期套餐", async () => {
      const form = createPackageForm({ period: "yearly", periodDays: 365, status: 0 });
      await PackageAPI.add(form);

      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      expect(created).toBeDefined();
      expect(created?.period).toBe("yearly");
      expect(created?.periodDays).toBe(365);

      if (created?.id) createdPackageIds.push(created.id);
    });

    test("异常：缺少必填字段 name", async () => {
      const form = createPackageForm();
      const { name, ...rest } = form;
      await expectBizError(PackageAPI.add(rest as any), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("异常：价格为负数", async () => {
      const form = createPackageForm({ salePrice: -1 });
      await expectBizError(PackageAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("GET /api/v1/packages/{id}/form - 获取套餐表单数据", () => {
    let testPackageId: number;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createPackageForm({ status: 0 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      if (!created?.id) throw new Error("未能创建测试套餐");
      testPackageId = created.id;
      createdPackageIds.push(testPackageId);
    });

    test("正向测试：获取表单数据", async () => {
      const form = await PackageAPI.getForm(testPackageId);

      expect(form).toBeDefined();
      expect(form.id).toBe(testPackageId);
      expect(form.name).toBeTruthy();
      expect(form.levelCode).toBeDefined();
      expect(form.period).toBeDefined();
      expect(typeof form.originalPrice).toBe("number");
      expect(typeof form.salePrice).toBe("number");
    });

    test("异常：套餐不存在", async () => {
      await expectBizError(PackageAPI.getForm(99999999), ["A0520", "A0400", "ERR_BAD_REQUEST"]);
    });
  });

  describe("PUT /api/v1/packages/{id} - 修改套餐", () => {
    let testPackageId: number;
    let originalForm: any;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createPackageForm({ status: 0 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      if (!created?.id) throw new Error("未能创建测试套餐");
      testPackageId = created.id;
      createdPackageIds.push(testPackageId);
      originalForm = await PackageAPI.getForm(testPackageId);
    });

    test("正向测试：修改套餐名称", async () => {
      const newName = `更新套餐_${Date.now()}`;
      const form = { ...originalForm, name: newName };
      await PackageAPI.update(testPackageId, form);

      const updated = await PackageAPI.getForm(testPackageId);
      expect(updated.name).toBe(newName);
    });

    test("正向测试：修改套餐价格", async () => {
      const newSalePrice = originalForm.originalPrice;
      const form = { ...originalForm, salePrice: newSalePrice };
      await PackageAPI.update(testPackageId, form);

      const updated = await PackageAPI.getForm(testPackageId);
      expect(updated.salePrice).toBe(newSalePrice);
    });

    test("异常：套餐不存在", async () => {
      const form = { ...originalForm };
      await expectBizError(PackageAPI.update(99999999, form), [
        "A0520",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/packages/{id}/status - 上架/下架", () => {
    let testPackageId: number;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createPackageForm({ status: 0 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      if (!created?.id) throw new Error("未能创建测试套餐");
      testPackageId = created.id;
      createdPackageIds.push(testPackageId);
    });

    test("正向测试：上架套餐", async () => {
      await PackageAPI.updateStatus(testPackageId, 1);

      const form = await PackageAPI.getForm(testPackageId);
      expect(form.status).toBe(1);
    });

    test("正向测试：下架套餐", async () => {
      await PackageAPI.updateStatus(testPackageId, 0);

      const form = await PackageAPI.getForm(testPackageId);
      expect(form.status).toBe(0);
    });

    test("异常：套餐不存在", async () => {
      await expectBizError(PackageAPI.updateStatus(99999999, 1), [
        "A0520",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/packages/{ids} - 删除套餐", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：删除单个套餐", async () => {
      const form = createPackageForm({ status: 0 });
      await PackageAPI.add(form);
      const page = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const created = page.list.find((p) => p.name === form.name);
      expect(created?.id).toBeDefined();
      const packageId = created!.id!;

      await PackageAPI.deleteByIds(packageId.toString());

      const pageAfter = await PackageAPI.getPage(createPackageQuery({ name: form.name }));
      const found = pageAfter.list.find((p) => p.id === packageId);
      expect(found).toBeUndefined();
    });

    test("异常：套餐不存在", async () => {
      await expectBizError(PackageAPI.deleteByIds("99999999"), [
        "A0520",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/packages/sales/stats - 销售统计", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：获取销售统计数据结构", async () => {
      const stats = await PackageAPI.getSalesStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalSales).toBe("number");
      expect(typeof stats.totalRevenue).toBe("number");
      expect(Array.isArray(stats.packageStats)).toBe(true);
      expect(Array.isArray(stats.levelStats)).toBe(true);
      expect(Array.isArray(stats.periodStats)).toBe(true);
      expect(stats.couponStats).toBeDefined();
      expect(typeof stats.couponStats.totalIssued).toBe("number");
      expect(typeof stats.couponStats.totalUsed).toBe("number");
    });
  });

  // ============ 优惠券管理 ============

  describe("CouponAPI - 后台优惠券管理（admin）", () => {
    describe("POST /api/v1/packages/coupons - 创建优惠券", () => {
      beforeAll(async () => {
        await login(USERS.ADMIN.username);
      });

      test("正向测试：创建满减券", async () => {
        const form = createCouponForm({ type: "full_reduction" });
        const result = await CouponAPI.add(form);

        expect(result).toBeDefined();
        expect(result.id).toBeGreaterThan(0);
        createdCouponIds.push(result.id);
      });

      test("正向测试：创建折扣券", async () => {
        const form = createCouponForm({ type: "discount", faceValue: 85 });
        const result = await CouponAPI.add(form);

        expect(result).toBeDefined();
        expect(result.id).toBeGreaterThan(0);
        createdCouponIds.push(result.id);
      });

      test("正向测试：创建无门槛券", async () => {
        const form = createCouponForm({
          type: "no_threshold",
          faceValue: 500,
        });
        delete (form as any).threshold;
        const result = await CouponAPI.add(form);

        expect(result).toBeDefined();
        expect(result.id).toBeGreaterThan(0);
        createdCouponIds.push(result.id);
      });

      test("异常：缺少必填字段 name", async () => {
        const form = createCouponForm();
        const { name, ...rest } = form;
        await expectBizError(CouponAPI.add(rest as any), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
      });

      test("异常：库存为负数", async () => {
        const form = createCouponForm({ totalQty: -2 });
        await expectBizError(CouponAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
      });
    });

    describe("GET /api/v1/packages/coupons/page - 优惠券分页列表", () => {
      beforeAll(async () => {
        await login(USERS.ADMIN.username);
      });

      test("正向测试：分页查询", async () => {
        const result = await CouponAPI.getPage(createCouponQuery({ pageNum: 1, pageSize: 10 }));

        expect(result).toBeDefined();
        expect(Array.isArray(result.list)).toBe(true);
        expect(typeof result.total).toBe("number");

        for (const c of result.list) {
          expect(c.id).toBeGreaterThan(0);
          expect(c.name).toBeTruthy();
          expect(["full_reduction", "discount", "no_threshold", "trial"]).toContain(c.type);
          expect(typeof c.faceValue).toBe("number");
          expect(typeof c.totalQty).toBe("number");
          expect(typeof c.issuedQty).toBe("number");
          expect(typeof c.usedQty).toBe("number");
          expect(c.issuedQty).toBeLessThanOrEqual(c.totalQty);
        }
      });

      test("正向测试：按类型筛选", async () => {
        const result = await CouponAPI.getPage(
          createCouponQuery({ type: "full_reduction", pageNum: 1, pageSize: 10 })
        );
        for (const c of result.list) {
          expect(c.type).toBe("full_reduction");
        }
      });

      describe("PUT /api/v1/packages/coupons/{id} - 修改优惠券", () => {
        let testCouponId: number;
        let originalForm: any;

        beforeAll(async () => {
          await login(USERS.ADMIN.username);
          const form = createCouponForm();
          const result = await CouponAPI.add(form);
          testCouponId = result.id;
          createdCouponIds.push(testCouponId);

          const page = await CouponAPI.getPage(createCouponQuery({ name: form.name }));
          const created = page.list.find((c) => c.id === testCouponId);
          originalForm = {
            ...form,
            id: testCouponId,
            status: created?.status ?? 1,
          };
        });

        test("正向测试：修改优惠券名称", async () => {
          const newName = `更新券_${Date.now()}`;
          const form = { ...originalForm, name: newName };
          await CouponAPI.update(testCouponId, form);

          const page = await CouponAPI.getPage(createCouponQuery({ name: newName }));
          const found = page.list.find((c) => c.id === testCouponId);
          expect(found?.name).toBe(newName);
        });

        test("异常：优惠券不存在", async () => {
          const form = { ...originalForm, id: 99999999 };
          await expectBizError(CouponAPI.update(99999999, form), [
            "A0523",
            "A0400",
            "ERR_BAD_REQUEST",
          ]);
        });
      });
    });

    describe("DELETE /api/v1/packages/coupons/{ids} - 删除优惠券", () => {
      beforeAll(async () => {
        await login(USERS.ADMIN.username);
      });

      test("正向测试：删除单个优惠券", async () => {
        const form = createCouponForm();
        const result = await CouponAPI.add(form);
        const couponId = result.id;

        await CouponAPI.deleteByIds(couponId.toString());

        const page = await CouponAPI.getPage(createCouponQuery({ name: form.name }));
        const found = page.list.find((c) => c.id === couponId);
        expect(found).toBeUndefined();
      });

      test("异常：优惠券不存在", async () => {
        await expectBizError(CouponAPI.deleteByIds("99999999"), [
          "A0523",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      });
    });

    describe("POST /api/v1/packages/coupons/batch - 批量发放", () => {
      let testCouponId: number;

      beforeAll(async () => {
        await login(USERS.ADMIN.username);
        const form = createCouponForm({ totalQty: 1000, perUserLimit: 10 });
        const result = await CouponAPI.add(form);
        testCouponId = result.id;
        createdCouponIds.push(testCouponId);
      });

      test("正向测试：按等级发放", async () => {
        const result = await CouponAPI.batchDistribute({
          couponId: testCouponId,
          targetScope: "level",
          levelCodes: ["level_0", "level_1"],
        });

        expect(result).toBeDefined();
        expect(typeof result.successCount).toBe("number");
        expect(typeof result.failCount).toBe("number");
        expect(result.successCount + result.failCount).toBeGreaterThan(0);
      });

      test("正向测试：按用户发放", async () => {
        const result = await CouponAPI.batchDistribute({
          couponId: testCouponId,
          targetScope: "users",
          userIds: [USERS.USER.id],
        });

        expect(result).toBeDefined();
        expect(result.successCount).toBeGreaterThan(0);
      });

      test("异常：优惠券不存在", async () => {
        await expectBizError(
          CouponAPI.batchDistribute({
            couponId: 99999999,
            targetScope: "users",
            userIds: [USERS.USER.id],
          }),
          ["A0523", "A0400", "ERR_BAD_REQUEST"]
        );
      });
    });
  });

  // ============ 用户端优惠券接口（使用 user 账号） ============

  describe("CouponAPI - 用户端优惠券操作（user）", () => {
    let testCouponId: number;

    beforeAll(async () => {
      // admin 创建优惠券供 user 领取
      await login(USERS.ADMIN.username);
      const form = createCouponForm({ totalQty: 100, perUserLimit: 5 });
      const result = await CouponAPI.add(form);
      testCouponId = result.id;
      createdCouponIds.push(testCouponId);
      // 切到 user 测试领取
      await login(userAccount);
    });

    describe("POST /api/v1/packages/coupons/{id}/receive - 领取优惠券", () => {
      test("正向测试：领取优惠券", async () => {
        const result = await CouponAPI.receive(testCouponId);

        expect(result).toBeDefined();
        expect(result.userCouponId).toBeGreaterThan(0);
      });

      test("异常：优惠券不存在", async () => {
        await expectBizError(CouponAPI.receive(99999999), ["A0523", "A0400", "ERR_BAD_REQUEST"]);
      });
    });

    describe("GET /api/v1/packages/coupons/my - 我的优惠券列表", () => {
      test("正向测试：获取我的优惠券列表", async () => {
        const list = await CouponAPI.listMy();

        expect(Array.isArray(list)).toBe(true);
        for (const c of list) {
          expect(c.id).toBeGreaterThan(0);
          expect(c.couponId).toBeGreaterThan(0);
          expect(c.couponName).toBeTruthy();
          expect([1, 2, 3, 4]).toContain(c.status);
          expect(c.receiveTime).toBeTruthy();
        }
      });

      test("正向测试：按状态筛选", async () => {
        const list = await CouponAPI.listMy(1);
        for (const c of list) {
          expect(c.status).toBe(1);
        }
      });
    });
  });
});
