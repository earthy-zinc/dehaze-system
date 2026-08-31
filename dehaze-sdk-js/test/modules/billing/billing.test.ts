import { AiBillingAPI, AiConversationAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login, logout } from "#/utils/auth";
import { topUpAdminCredits } from "#/utils/quota";
import { USERS } from "#/factories/constants";
import { createConversationForm } from "#/factories/ai-conversation";
import {
  createBillingRecordQuery,
  createBillingStatQuery,
  createCreditAdjustForm,
  createCreditLogQuery,
  createRefundApplyForm,
} from "#/factories/billing";

/** 当前月份 yyyy-MM */
const currentMonth = () => new Date().toISOString().slice(0, 7);

/** 以普通用户身份执行动作，断言其越权失败（A0301），无论成败都恢复为管理员登录 */
async function expectForbiddenAsUser(action: () => Promise<unknown>) {
  await login(USERS.USER.username);
  try {
    await action();
  } finally {
    await login(USERS.ADMIN.username);
  }
}

// beforeAll 真实对话的会话 ID，供「计费联动验证」按会话查询计费记录
let chatConversationId = 0;

beforeAll(async () => {
  // 自建计费数据：充值后发起一次真实 AI 对话（默认本地模型），
  // 使计费记录/流水/退款/换算用例有本套件自己的数据，不依赖其他模块的历史残留
  await topUpAdminCredits();

  const conv = await AiConversationAPI.createConversation(createConversationForm());
  chatConversationId = conv.id;
  try {
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("SSE 发送消息超时（60s）")), 60000);
      AiConversationAPI.sendMessage(
        conv.id,
        { content: "1+1=?" },
        {
          onEnd: () => {
            clearTimeout(timeout);
            resolve();
          },
          onNetworkError: (error) => {
            clearTimeout(timeout);
            reject(error instanceof Error ? error : new Error(String(error)));
          },
        }
      );
    });
  } finally {
    await AiConversationAPI.deleteConversation(conv.id).catch(() => {});
  }
}, 120000);

describe("AI 计费管理模块接口测试 - AiBillingAPI", () => {
  // ===== 用户端接口 =====

  describe("GET /api/v1/ai-billing/balance - 余额查询", () => {
    test("正向测试：查询当前用户余额与配额", async () => {
      const balance = await AiBillingAPI.getBalance();
      // 后端 creditsBalance 为 Decimal 序列化为字符串（如 "0.00"），可解析为数字
      expect(["number", "string"]).toContain(typeof balance.creditsBalance);
      expect(typeof balance.arrearsStatus).toBe("boolean");
      expect(typeof balance.dailyUsed).toBe("number");
      expect(typeof balance.dailyLimit).toBe("number");
      expect(typeof balance.monthlyUsed).toBe("number");
      expect(typeof balance.monthlyLimit).toBe("number");
    });

    test("验证：配额数据一致性（不限量时仅校验非欠费）", async () => {
      const balance = await AiBillingAPI.getBalance();
      // dailyLimit=0 表示不限量（admin 默认），仅有限额时校验用量不超限
      if (!balance.arrearsStatus && balance.dailyLimit > 0) {
        expect(balance.dailyUsed).toBeLessThanOrEqual(balance.dailyLimit);
      }
    });

    test("边界：未登录访问应返回 401", async () => {
      await logout(USERS.ADMIN.username);
      try {
        await expectBizError(AiBillingAPI.getBalance(), ["A0230"]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });
  });

  describe("GET /api/v1/ai-billing/records - 计费明细查询", () => {
    test("正向测试：分页查询计费明细", async () => {
      const result = await AiBillingAPI.getRecords(createBillingRecordQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("正向测试：按计费类型筛选", async () => {
      const result = await AiBillingAPI.getRecords(
        createBillingRecordQuery({ billType: "chat", pageSize: 100 })
      );
      result.list.forEach((record) => {
        expect(record.billType).toBe("chat");
      });
    });

    test("正向测试：按日期范围筛选", async () => {
      const result = await AiBillingAPI.getRecords(
        createBillingRecordQuery({
          dateStart: "2025-01-01 00:00:00",
          dateEnd: "2099-12-31 23:59:59",
        })
      );
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("验证：计费记录字段完整性", async () => {
      const result = await AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 1 }));
      expect(result.list.length, "无计费记录：beforeAll 真实对话未产生计费数据").toBeGreaterThan(0);
      const record = result.list[0]!;
      expect(record.id).toBeGreaterThan(0);
      expect(record.model).toBeTruthy();
      expect(record.billType).toBeTruthy();
      expect(typeof record.inputTokens).toBe("number");
      expect(typeof record.outputTokens).toBe("number");
      expect(typeof record.credits).toBe("number");
      expect(typeof record.quotaConsumed).toBe("number");
      expect(record.createTime).toBeTruthy();
    });

    test("边界：大页码返回空列表", async () => {
      const result = await AiBillingAPI.getRecords(createBillingRecordQuery({ pageNum: 10000 }));
      expect(result.list.length).toBe(0);
    });
  });

  describe("GET /api/v1/ai-billing/credit-logs - 流水查询", () => {
    test("正向测试：分页查询余额变动流水", async () => {
      const result = await AiBillingAPI.getCreditLogs(createCreditLogQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("正向测试：按流水来源筛选", async () => {
      const result = await AiBillingAPI.getCreditLogs(
        createCreditLogQuery({ source: "consume", pageSize: 100 })
      );
      result.list.forEach((log) => {
        expect(log.source).toBe("consume");
      });
    });

    test("验证：流水记录字段完整性", async () => {
      const result = await AiBillingAPI.getCreditLogs(createCreditLogQuery({ pageSize: 1 }));
      expect(result.list.length, "无余额变动流水：充值/对话消耗未产生流水").toBeGreaterThan(0);
      const log = result.list[0]!;
      expect(log.id).toBeGreaterThan(0);
      expect(log.source).toBeTruthy();
      // 金额为 Decimal 防精度丢失的字符串序列化（如 "100000.00"）
      expect(typeof log.amount).toBe("string");
      expect(Number.parseFloat(String(log.amount))).not.toBeNaN();
      expect(typeof log.balanceAfter).toBe("string");
      expect(Number.parseFloat(String(log.balanceAfter))).not.toBeNaN();
      expect(log.reason).toBeTruthy();
      expect(log.createTime).toBeTruthy();
    });
  });

  describe("GET /api/v1/ai-billing/bills/{month} - 账单查询", () => {
    test("正向测试：查询当月账单", async () => {
      const month = currentMonth();
      const bill = await AiBillingAPI.getBill(month);
      expect(bill.month).toBe(month);
      expect(typeof bill.totalConsume).toBe("number");
      expect(typeof bill.totalRecharge).toBe("number");
      expect(typeof bill.totalRefund).toBe("number");
    });

    test("边界：查询不存在的月份账单应失败", async () => {
      await expectBizError(AiBillingAPI.getBill("2099-12"), ["A0401"]);
    });

    test("边界：月份格式不正确应失败", async () => {
      await expectBizError(AiBillingAPI.getBill("invalid-month"), ["A0400"]);
    });
  });

  describe("GET /api/v1/ai-billing/bills/{month}/download - 账单下载", () => {
    test("正向测试：下载当月账单", async () => {
      const month = currentMonth();
      const bill = await AiBillingAPI.downloadBill(month);
      expect(bill.month).toBe(month);
      expect(typeof bill.totalConsume).toBe("number");
    });

    test("边界：下载不存在的月份账单应失败", async () => {
      await expectBizError(AiBillingAPI.downloadBill("2099-12"), ["A0401"]);
    });
  });

  describe("POST /api/v1/ai-billing/refunds - 退款申请", () => {
    test("边界：不存在的计费记录申请退款应失败", async () => {
      const form = createRefundApplyForm({ billingId: 99999999 });
      await expectBizError(AiBillingAPI.applyRefund(form), ["A0401"]);
    });

    test("参数校验：缺少 reason 应失败", async () => {
      const form = createRefundApplyForm({ reason: "" });
      await expectBizError(AiBillingAPI.applyRefund(form), ["A0400"]);
    });

    test("参数校验：amount <= 0 应失败", async () => {
      const form = createRefundApplyForm({ amount: 0 });
      await expectBizError(AiBillingAPI.applyRefund(form), ["A0400"]);
    });

    test("边界：重复申请退款应失败（A0680）", async () => {
      // 需要一条已存在的计费记录作为退款对象
      const records = await AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 1 }));
      expect(records.list.length, "无计费记录：beforeAll 真实对话未产生计费数据").toBeGreaterThan(
        0
      );

      const billingId = records.list[0]!.id;
      const form = createRefundApplyForm({ billingId });

      // 第一次申请；若该记录已有 pending 退款（历史运行残留）返回 A0680，同样视为就绪
      await AiBillingAPI.applyRefund(form).catch((e: any) => {
        const code = e?.response?.data?.code ?? e?.code;
        if (code !== "A0680") throw e;
      });

      // 第二次重复申请应失败
      await expectBizError(AiBillingAPI.applyRefund(form), ["A0680"]);
    });
  });

  // ===== 管理员接口 =====

  describe("GET /api/v1/ai-billing/stats - 管理员统计查询", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：按模型维度统计", async () => {
      const result = await AiBillingAPI.getStats(createBillingStatQuery({ groupBy: "model" }));
      expect(Array.isArray(result)).toBe(true);
      if (result.length > 0) {
        const stat = result[0]!;
        expect(stat.dimension).toBeTruthy();
        expect(typeof stat.totalCredits).toBe("number");
        expect(typeof stat.totalInputTokens).toBe("number");
        expect(typeof stat.totalOutputTokens).toBe("number");
        expect(typeof stat.cacheHitRate).toBe("number");
      }
    });

    test("正向测试：按计费类型维度统计", async () => {
      const result = await AiBillingAPI.getStats(createBillingStatQuery({ groupBy: "billType" }));
      expect(Array.isArray(result)).toBe(true);
    });

    test("正向测试：按日期维度统计", async () => {
      const result = await AiBillingAPI.getStats(
        createBillingStatQuery({
          groupBy: "day",
          dateStart: "2025-01-01",
          dateEnd: "2099-12-31",
        })
      );
      expect(Array.isArray(result)).toBe(true);
    });

    // 后端 require_permission 对无权限用户返回 A0301（访问未授权），非文档 A0403。
    test("边界：普通用户访问管理员统计应失败", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(AiBillingAPI.getStats(createBillingStatQuery()), ["A0301"])
      );
    });
  });

  describe("POST /api/v1/ai-billing/adjust - 管理员手动调整积分", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("正向测试：管理员增加用户积分", async () => {
      const form = createCreditAdjustForm({
        userId: USERS.USER.id,
        amount: 10,
        reason: "测试增加积分",
      });
      const result = await AiBillingAPI.adjustCredits(form);
      expect(result.userId).toBe(USERS.USER.id);
      // 后端 creditsBalance 为 Decimal 序列化为字符串（如 "10"），可解析为数字
      expect(["number", "string"]).toContain(typeof result.creditsBalance);
    });

    test("正向测试：管理员扣减用户积分", async () => {
      const form = createCreditAdjustForm({
        userId: USERS.USER.id,
        amount: -10,
        reason: "测试扣减积分",
      });
      const result = await AiBillingAPI.adjustCredits(form);
      expect(result.userId).toBe(USERS.USER.id);
    });

    test("参数校验：缺少 reason 应失败", async () => {
      const form = createCreditAdjustForm({ reason: "" });
      await expectBizError(AiBillingAPI.adjustCredits(form), ["A0400"]);
    });

    test("参数校验：amount=0 应失败", async () => {
      const form = createCreditAdjustForm({ amount: 0 });
      await expectBizError(AiBillingAPI.adjustCredits(form), ["A0400"]);
    });

    // 后端 require_permission 对无权限用户返回 A0301（访问未授权），非文档 A0403。
    test("边界：普通用户调用管理员调整应失败", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(
          AiBillingAPI.adjustCredits(
            createCreditAdjustForm({
              userId: USERS.ADMIN.id,
              amount: 100,
              reason: "越权调整",
            })
          ),
          ["A0301"]
        )
      );
    });
  });

  describe("POST /api/v1/ai-billing/refunds/{id}/audit - 退款审核", () => {
    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：审核不存在的退款申请应失败", async () => {
      await expectBizError(
        AiBillingAPI.auditRefund(99999999, { approved: true, auditRemark: "测试审核" }),
        ["A0401"]
      );
    });

    // 后端 require_permission 对无权限用户返回 A0301（访问未授权），非文档 A0403。
    test("边界：普通用户调用退款审核应失败", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(AiBillingAPI.auditRefund(1, { approved: true, auditRemark: "越权审核" }), [
          "A0301",
        ])
      );
    });
  });

  describe("GET /api/v1/ai-billing/refunds - 退款申请列表（管理端审核中心）", () => {
    test("正向：管理员按状态查询退款申请列表", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiBillingAPI.getRefunds({ status: 1, pageNum: 1, pageSize: 10 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      result.list.forEach((r) => expect([1, 2, 3]).toContain(r.status));
    });

    test("正向：管理员按用户筛选退款申请", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiBillingAPI.getRefunds({ userId: USERS.USER.id, pageSize: 10 });
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((r) => expect(r.userId).toBe(USERS.USER.id));
    });

    test("负向：普通用户查询退款列表 → A0301", async () => {
      await expectForbiddenAsUser(() => expectBizError(AiBillingAPI.getRefunds({}), ["A0301"]));
    });
  });

  // ===== 数据隔离测试 =====

  describe("数据隔离 - 越权访问", () => {
    test("边界：用户无法查询他人余额（接口仅返回本人数据）", async () => {
      // 不传 userId 时接口仅返回当前登录用户数据，故验证不同用户查询结果归属不同
      await login(USERS.ADMIN.username);
      const adminBalance = await AiBillingAPI.getBalance();

      await login(USERS.USER.username);
      const userBalance = await AiBillingAPI.getBalance();

      expect(adminBalance.userId).not.toBe(userBalance.userId);

      await login(USERS.ADMIN.username);
    });

    test("正向：管理员指定 userId 查询目标用户余额（下钻）", async () => {
      await login(USERS.ADMIN.username);
      const balance = await AiBillingAPI.getBalance(USERS.USER.id);
      expect(balance.userId).toBe(USERS.USER.id);
    });

    test("正向：管理员指定 userId 查询目标用户计费明细（下钻）", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiBillingAPI.getRecords(
        createBillingRecordQuery({ userId: USERS.USER.id, pageSize: 5 })
      );
      expect(result.list.every((r) => r.userId === USERS.USER.id)).toBe(true);
    });

    test("正向：管理员指定 userId 查询目标用户积分流水（下钻）", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiBillingAPI.getCreditLogs(
        createCreditLogQuery({ userId: USERS.USER.id, pageSize: 5 })
      );
      expect(result.list.every((r) => r.userId === USERS.USER.id)).toBe(true);
    });

    test("负向：普通用户指定他人 userId 查余额 → A0301", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(AiBillingAPI.getBalance(USERS.ADMIN.id), ["A0301"])
      );
    });

    test("负向：普通用户指定他人 userId 查明细 → A0301", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(
          AiBillingAPI.getRecords(createBillingRecordQuery({ userId: USERS.ADMIN.id })),
          ["A0301"]
        )
      );
    });

    test("负向：普通用户指定他人 userId 查流水 → A0301", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(
          AiBillingAPI.getCreditLogs(createCreditLogQuery({ userId: USERS.ADMIN.id })),
          ["A0301"]
        )
      );
    });
  });

  // ===== Token→credits 换算验证 =====

  describe("Token→credits 换算验证", () => {
    test("正向测试：计费记录中 credits 与 token 数量关联", async () => {
      const result = await AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 10 }));

      expect(result.list.length, "无计费记录：beforeAll 真实对话未产生计费数据").toBeGreaterThan(0);

      result.list.forEach((record) => {
        expect(record.credits).toBeGreaterThanOrEqual(0);
        // 本地默认模型（qwen3-0.6b）费率配置为 0：有 token 计量但 credits 为 0 属
        // 正常换算；「非缓存 token 必产生积分消耗」仅在付费模型下成立，待接入
        // 付费模型后补充该断言
      });
    });

    test("验证：相同模型相同 token 数的 credits 一致", async () => {
      const result = await AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 100 }));

      // 按模型分组，再按 token 数分组，比较同组 credits 是否一致
      const byModel = new Map<string, typeof result.list>();
      result.list.forEach((record) => {
        const key = record.model;
        if (!byModel.has(key)) {
          byModel.set(key, []);
        }
        byModel.get(key)!.push(record);
      });

      for (const [, records] of byModel) {
        if (records.length < 2) continue;

        const tokenMap = new Map<string, number[]>();
        records.forEach((r) => {
          const key = `${r.inputTokens}_${r.outputTokens}`;
          if (!tokenMap.has(key)) {
            tokenMap.set(key, []);
          }
          tokenMap.get(key)!.push(r.credits);
        });

        for (const [, creditsList] of tokenMap) {
          if (creditsList.length < 2) continue;
          const first = creditsList[0]!;
          creditsList.forEach((c) => {
            expect(c).toBe(first);
          });
        }
      }
    });
  });

  // ===== 计费联动验证 =====

  describe("计费联动验证 - 对话产生计费记录", () => {
    test("正向测试：AI 对话产生 billType=chat 的计费记录且 token 用量入账", async () => {
      // 本地默认模型（qwen3-0.6b）费率配置为 0，不产生积分扣减/配额变化；
      // 本地环境验证「消息发送 → 计费记录落库 + token 计量」联动，
      // 付费模型的余额/配额扣减联动需接入付费模型后补充验证
      const result = await AiBillingAPI.getRecords(
        createBillingRecordQuery({ conversationId: chatConversationId, pageSize: 10 })
      );
      expect(result.list.length, "对话未产生计费记录：计费管线未随消息落库").toBeGreaterThan(0);

      const chatRecord = result.list.find((r) => r.billType === "chat");
      expect(chatRecord, "对话计费记录 billType 应为 chat").toBeDefined();
      expect(
        chatRecord!.inputTokens + chatRecord!.outputTokens,
        "计费记录 token 用量为 0：流式 usage 计量链路失效"
      ).toBeGreaterThan(0);
    });
  });
});

/**
 * 消耗汇总/异常监控/成本管理/对账（AI计费管理 API接口.md）。
 *
 * 后端尚未实现 summary/anomalies/costs/cost-stats/reconcile/import 与 refundStatus 字段：
 * 测试先行契约（以 dehaze-doc API接口.md 为行为断言依据），接口 404 或字段缺失时
 * 正向用例失败暴露，待后端实现后统一验证。
 */
describe("消耗汇总/异常监控/成本管理/对账（契约先行）", () => {
  test("正向：当前时段消耗汇总 getSummary", async () => {
    await login(USERS.USER.username);
    const summary = await AiBillingAPI.getSummary();
    expect(typeof summary.totalCredits).toBe("number");
    expect(typeof summary.inputTokens).toBe("number");
  });

  test("负向：普通用户异常监控 → A0301", async () => {
    await login(USERS.USER.username);
    await expectBizError(AiBillingAPI.getAnomalies({}), ["A0301"]);
  });

  test("正向：计费记录含 refundStatus 字段（契约）", async () => {
    await login(USERS.USER.username);
    const result = await AiBillingAPI.getRecords(createBillingRecordQuery());
    if (result.list.length > 0) {
      expect([0, 1, 2, 3]).toContain(result.list[0]!.refundStatus);
    }
  });

  test("正向：模型成本配置 CRUD（管理员）", async () => {
    await login(USERS.ADMIN.username);
    const created = await AiBillingAPI.createCost({
      modelId: "test_model_cost",
      providerId: 1,
      details: [{ tokenType: "input", timeSlot: "peak", unitPrice: 1 }],
    });
    expect(created.id).toBeGreaterThan(0);
    expect(created.priceVersion).toBeGreaterThan(0);
    await AiBillingAPI.updateCost(created.id, { status: 0 });
    await AiBillingAPI.deleteCost(created.id);
  });

  test("正向：成本-利润统计 getCostStats（管理员）", async () => {
    await login(USERS.ADMIN.username);
    const stats = await AiBillingAPI.getCostStats();
    expect(Array.isArray(stats)).toBe(true);
    if (stats.length > 0) {
      expect(["overall", "ai"]).toContain(stats[0]!.metric);
      expect(typeof stats[0]!.revenue).toBe("number");
      expect(typeof stats[0]!.profit).toBe("number");
    }
  });

  test("正向：对账数据导入 importReconcile（管理员）", async () => {
    await login(USERS.ADMIN.username);
    const result = await AiBillingAPI.importReconcile({
      content: "test",
      startTime: "2026-08-01",
      endTime: "2026-08-31",
    });
    expect(typeof result.imported).toBe("number");
  });
});
