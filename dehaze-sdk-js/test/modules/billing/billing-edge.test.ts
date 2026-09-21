import { AiBillingAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createBillingRecordQuery,
  createCreditAdjustForm,
  createCreditLogQuery,
} from "#/factories/billing";

/** 以普通用户身份执行动作，断言其越权失败（A0301），无论成败都恢复为管理员登录 */
async function expectForbiddenAsUser(action: () => Promise<unknown>) {
  await login(USERS.USER.username);
  try {
    await action();
  } finally {
    await login(USERS.ADMIN.username);
  }
}

describe("AI 计费管理 - 边界与对抗性强化 billing-edge", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  // ===== 参数边界 =====

  describe("分页参数边界（A0400）", () => {
    test("records pageNum=0 拒绝", async () => {
      await expectBizError(AiBillingAPI.getRecords(createBillingRecordQuery({ pageNum: 0 })), [
        "A0400",
      ]);
    });

    test("records pageSize=101 拒绝（上限 100）", async () => {
      await expectBizError(AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 101 })), [
        "A0400",
      ]);
    });

    test("credit-logs pageSize=101 拒绝", async () => {
      await expectBizError(AiBillingAPI.getCreditLogs(createCreditLogQuery({ pageSize: 101 })), [
        "A0400",
      ]);
    });

    test("refunds pageSize=101 拒绝", async () => {
      await expectBizError(AiBillingAPI.getRefunds({ pageSize: 101 }), ["A0400"]);
    });

    test("balance userId=0（ge=1 下界）拒绝", async () => {
      await expectBizError(AiBillingAPI.getBalance(0), ["A0400"]);
    });
  });

  describe("summary / bills 参数边界", () => {
    test("正向：day / month 维度均可用", async () => {
      const day = await AiBillingAPI.getSummary("day");
      expect(typeof day.totalCredits).toBe("number");
      const month = await AiBillingAPI.getSummary("month");
      expect(typeof month.totalCredits).toBe("number");
    });

    test("负向：非法维度 year → A0400", async () => {
      await expectBizError(AiBillingAPI.getSummary("year" as any), ["A0400"]);
    });

    test("负向：账单月份 2099-13 非法 → A0400", async () => {
      await expectBizError(AiBillingAPI.getBill("2099-13"), ["A0400"]);
    });

    test("边界：非零填充月份 2026-1 被 strptime 接受，空账期 → A0401", async () => {
      await expectBizError(AiBillingAPI.downloadBill("2026-1"), ["A0401"]);
    });

    test("负向：records 非法日期格式严格拒绝 → A0400（不静默忽略）", async () => {
      await expectBizError(
        AiBillingAPI.getRecords(createBillingRecordQuery({ dateStart: "not-a-date" })),
        ["A0400"]
      );
    });
  });

  // ===== 对抗性脏语料 =====

  describe("管理员调整 reason 对抗性脏语料原样落流水", () => {
    test("emoji + 零宽 + CRLF + 全半角混杂 + 超长 reason 原样写入流水", async () => {
      const unique = `边缘${Date.now()}`;
      const dirty = `手动调整${unique}\r\n零宽​全角ＡＢｃ半角ABC\U0001F600` + "备注补充" * 20;

      const result = await AiBillingAPI.adjustCredits(
        createCreditAdjustForm({
          userId: USERS.USER.id,
          amount: 1,
          reason: dirty,
        })
      );
      expect(result.userId).toBe(USERS.USER.id);

      const logs = await AiBillingAPI.getCreditLogs(
        createCreditLogQuery({ userId: USERS.USER.id, source: "admin_adjust", pageSize: 50 })
      );
      const matched = logs.list.filter((l) => l.reason === dirty);
      expect(
        matched.length,
        "脏语料 reason 必须原样写入流水（禁止截断/清洗/编码变形）"
      ).toBeGreaterThan(0);
      expect(Number(matched[0]!.amount)).toBe(1);
      expect(matched[0]!.operatorId).toBe(USERS.ADMIN.id);
    });

    test("参数校验：amount 非整数（1.5）→ A0400", async () => {
      await expectBizError(
        AiBillingAPI.adjustCredits(
          createCreditAdjustForm({ userId: USERS.USER.id, amount: 1.5, reason: "非整数" })
        ),
        ["A0400"]
      );
    });

    // 【暴露缺陷】adjust 对不存在的用户未校验存在性：balance CAS 读取 None 静默跳过，
    // 返回 200 并为幽灵 user_id 写入积分流水。期望 A0401（资源不存在）。
    test("负向：adjust 不存在的用户应拒绝（暴露性用例）", async () => {
      await expectBizError(
        AiBillingAPI.adjustCredits(
          createCreditAdjustForm({
            userId: 99999999,
            amount: 1,
            reason: "幽灵用户调整",
          })
        ),
        ["A0400", "A0401"]
      );
    });
  });

  // ===== 越权与成本数据隔离（T-AB-044f / T-AB-050）=====

  describe("成本数据与异常清单权限隔离", () => {
    test("负向：普通用户查成本单价列表 → A0301", async () => {
      await expectForbiddenAsUser(() => expectBizError(AiBillingAPI.getCosts({}), ["A0301"]));
    });

    test("负向：普通用户查成本-利润统计 → A0301", async () => {
      await expectForbiddenAsUser(() => expectBizError(AiBillingAPI.getCostStats(), ["A0301"]));
    });

    test("负向：普通用户导入对账数据 → A0301", async () => {
      await expectForbiddenAsUser(() =>
        expectBizError(
          AiBillingAPI.importReconcile({
            content: "row",
            startTime: "2026-08-01",
            endTime: "2026-08-31",
          }),
          ["A0301"]
        )
      );
    });

    test("负向：普通用户查异常计费清单 → A0301", async () => {
      await expectForbiddenAsUser(() => expectBizError(AiBillingAPI.getAnomalies({}), ["A0301"]));
    });
  });

  // ===== 性能烟测 =====

  describe("性能烟测", () => {
    test("getBalance < 1000ms", async () => {
      const start = Date.now();
      await AiBillingAPI.getBalance();
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000);
    });

    test("getRecords 分页查询 < 2000ms", async () => {
      const start = Date.now();
      await AiBillingAPI.getRecords(createBillingRecordQuery({ pageSize: 20 }));
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(2000);
    });
  });
});
