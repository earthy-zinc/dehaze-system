import { beforeEach, describe, expect, it, vi } from "vitest";
import AiBillingAPI from "@/api/ai-billing";
import type { AnomalyRecordQuery, BillVO, ImportReconcileForm } from "@/api/ai-billing/model";

/**
 * AI 计费管理 API 请求契约单元测试（mock axios 封装层，锁 URL/wire 字段）。
 *
 * 三处契约按 python（唯一行为事实源）的实际接收口径对齐：
 * - 异常记录查询支持 `userId`（router/ai_billing.py list_anomalies）；
 * - 成本版本新增 `providerId` 必填（schema/ai_billing_cost.py ModelCostCreateRequest.provider_id = Field(...)）；
 * - 对账导入起止时间可空（ReconcileImportRequest.start_time/end_time 默认 None）。
 */

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: requestMock }));

describe("AiBillingAPI 请求契约", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({});
  });

  it("getAnomalies 透传 userId 与分页参数", async () => {
    const query: AnomalyRecordQuery = { pageNum: 1, pageSize: 20, userId: 7, status: 0 };
    await AiBillingAPI.getAnomalies(query);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai-billing/anomalies",
      method: "get",
      params: query,
    });
  });

  it("createCost 提交 providerId，缺省在类型层被拒（python 必填字段）", async () => {
    await AiBillingAPI.createCost({ modelId: "qwen3-0.6b", providerId: 3, status: 1 });
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai-billing/costs",
      method: "post",
      data: { modelId: "qwen3-0.6b", providerId: 3, status: 1 },
    });

    // 负向守卫（判据在类型层，由 `npx tsc --noEmit` 验证）：providerId 改回可选会使本行报"未使用的 @ts-expect-error"。
    // @ts-expect-error providerId 为必填
    await AiBillingAPI.createCost({ modelId: "qwen3-0.6b" });
  });

  it("BillVO 余额为字符串（python/java Decimal 字符串口径，类型层守卫）", async () => {
    requestMock.mockResolvedValue({
      userId: 1,
      month: "2026-07",
      totalConsume: 100,
      totalRecharge: 0,
      totalRefund: 0,
      balanceStart: "3000.00",
      balanceEnd: "1500.00",
      itemSummary: { chat: 100, tool_llm: 0, kb_inject: 0, asr: 0, tts: 0, image: 0 },
    });

    const bill = await AiBillingAPI.getBill("2026-07");

    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai-billing/bills/2026-07",
      method: "get",
    });
    // 字符串原样可用于展示；参与计算须先 Number() 转换（后端防 Decimal 精度丢失）
    expect(bill.balanceStart).toBe("3000.00");
    expect(Number(bill.balanceEnd)).toBe(1500);

    const typed: BillVO = {
      userId: 1,
      month: "2026-07",
      totalConsume: 100,
      totalRecharge: 0,
      totalRefund: 0,
      balanceStart: "3000.00",
      balanceEnd: "1500.00",
      itemSummary: { chat: 100, tool_llm: 0, kb_inject: 0, asr: 0, tts: 0, image: 0 },
    };
    expect(Number(typed.balanceStart)).toBe(3000);

    // 负向守卫（判据在类型层，由 `npx tsc --noEmit` 验证）：余额必须是 string，
    // 改回 number 会使本行报"未使用的 @ts-expect-error"。
    // @ts-expect-error 余额为字符串，不得按数字字面量赋值
    const broken: BillVO = { ...typed, balanceStart: 3000 };
    expect(broken.balanceStart).toBe(3000);
  });

  it("importReconcile 允许只传 content（对账周期可空）", async () => {
    const form: ImportReconcileForm = { content: "qwen3-0.6b,1200000" };
    await AiBillingAPI.importReconcile(form);
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai-billing/reconcile/import",
      method: "post",
      data: form,
    });
  });
});
