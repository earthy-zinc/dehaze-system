import { beforeAll, describe, expect, test } from "vitest";
import { AiEvalAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";

/**
 * AI 评测中心（F-M08-014，跨 Agent 聚合）
 *
 * 全部端点只读且需 ai:agent:manage，不产生测试数据（复核详情/对比的负向用例除外）。
 * 断言锁定 camelCase 字段映射——后端响应经 OrmResult 别名转换，dict 内层键保持
 * snake_case，字段名回归在类型层面无法发现。
 *
 * 复核详情正向用例（样本定义 + 实际输出 + 四维得分）需要一条已完成评测及其样本
 * 执行结果，依赖真实模型可用性，故待 8991 服务与模型就绪后在评测域用例中补充。
 */
describe("AI 评测中心 - AiEvalAPI (F-M08-014)", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  test("正向：总览返回各 Agent 门禁状态/退化标识", async () => {
    const overview = await AiEvalAPI.getOverview();
    expect(Array.isArray(overview)).toBe(true);
    for (const item of overview) {
      expect(typeof item.agentId).toBe("number");
      expect(typeof item.agentCode).toBe("string");
      expect(["passed", "failed", "none"]).toContain(item.gateStatus);
      expect(typeof item.degraded).toBe("boolean");
      expect(typeof item.highRiskFailed).toBe("boolean");
    }
  });

  test("正向：趋势返回已完成评测的得分序列", async () => {
    const trends = await AiEvalAPI.getTrends({ limit: 10 });
    expect(Array.isArray(trends)).toBe(true);
    for (const item of trends) {
      expect(typeof item.runId).toBe("number");
      expect([2, 3]).toContain(item.status);
    }
  });

  test("正向：判分状态返回一致性状态与复核统计", async () => {
    const status = await AiEvalAPI.getJudgeStatus();
    expect(["normal", "drifted", "insufficient_data"]).toContain(status.consistencyState);
    expect(typeof status.driftPaused).toBe("boolean");
    expect(status.reviewStats.pending + status.reviewStats.reviewed).toBe(status.reviewStats.total);
  });

  test("正向：复核队列统计与列表长度一致", async () => {
    const queue = await AiEvalAPI.getReviews();
    expect(Array.isArray(queue.items)).toBe(true);
    expect(queue.pending + queue.reviewed).toBe(queue.items.length);
    for (const item of queue.items) {
      expect(typeof item.runId).toBe("number");
      expect(typeof item.sampleId).toBe("number");
      expect([1, 2]).toContain(item.status);
    }
  });

  test("负向：评测记录不存在 → A0401", async () => {
    await expectBizError(AiEvalAPI.getReviewDetail(999999999, 1), ["A0401"]);
  });

  test("负向：对比基准 run 不存在 → A0401", async () => {
    await expectBizError(AiEvalAPI.compareRuns(999999999, 1), ["A0401"]);
  });
});
