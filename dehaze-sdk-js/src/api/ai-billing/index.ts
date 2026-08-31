import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  AnomalyRecordQuery,
  AnomalyRecordVO,
  BalanceVO,
  BillVO,
  BillingRecordQuery,
  BillingRecordVO,
  BillingStatQuery,
  BillingStatVO,
  BillingRefundApplyForm,
  BillingRefundAuditForm,
  BillingRefundVO,
  BillingSummaryVO,
  CostStatQuery,
  CostStatVO,
  CreditAdjustForm,
  CreditLogQuery,
  CreditLogVO,
  ImportReconcileForm,
  BillingRefundQuery,
  ModelCostForm,
  ModelCostQuery,
  ModelCostVO,
} from "./model";

/**
 * AI 计费管理 API
 *
 * 余额查询、计费明细、流水、账单、退款申请（用户端）；
 * 统计、手动调整、退款审核（管理员端）。
 */
class AiBillingAPI {
  // ==================== 用户端接口 ====================

  /**
   * 查询余额与配额使用情况
   *
   * 传 userId 为管理端下钻查询指定用户，不传为当前登录用户。
   */
  static getBalance(userId?: number) {
    return request<BalanceVO>({
      url: "/api/v1/ai-billing/balance",
      method: "get",
      params: userId == null ? undefined : { userId },
    });
  }

  /** 当前时段消耗汇总（Token 使用量/积分费用），dimension：day-当日 / month-当月 */
  static getSummary(dimension?: "day" | "month") {
    return request<BillingSummaryVO>({
      url: "/api/v1/ai-billing/summary",
      method: "get",
      params: dimension ? { dimension } : undefined,
    });
  }

  /** 分页查询计费明细，query.userId 不传为当前登录用户 */
  static getRecords(query?: BillingRecordQuery) {
    return request<PageResult<BillingRecordVO[]>>({
      url: "/api/v1/ai-billing/records",
      method: "get",
      params: query,
    });
  }

  /** 分页查询余额变动流水，query.userId 不传为当前登录用户 */
  static getCreditLogs(query?: CreditLogQuery) {
    return request<PageResult<CreditLogVO[]>>({
      url: "/api/v1/ai-billing/credit-logs",
      method: "get",
      params: query,
    });
  }

  /** 查询指定月份的月结账单 */
  static getBill(month: string) {
    return request<BillVO>({
      url: `/api/v1/ai-billing/bills/${month}`,
      method: "get",
    });
  }

  /**
   * 下载指定月份的月结账单（PDF/Excel）
   *
   * 返回 Blob，前端可用 URL.createObjectURL 生成下载链接。
   */
  static downloadBill(month: string) {
    // 后端契约：账单下载返回 {code,msg,data} JSON 信封（data 为账单内容，前端可另存为文件）
    return request<BillVO>({
      url: `/api/v1/ai-billing/bills/${month}/download`,
      method: "get",
    });
  }

  /** 用户申请误扣退款 */
  static applyRefund(data: BillingRefundApplyForm) {
    return request<BillingRefundVO>({
      url: "/api/v1/ai-billing/refunds",
      method: "post",
      data,
    });
  }

  // ==================== 管理员接口 ====================

  /** 管理员多维度统计查询 */
  static getStats(query: BillingStatQuery) {
    return request<BillingStatVO[]>({
      url: "/api/v1/ai-billing/stats",
      method: "get",
      params: query,
    });
  }

  /** 管理员手动调整用户积分 */
  static adjustCredits(data: CreditAdjustForm) {
    return request<BalanceVO>({
      url: "/api/v1/ai-billing/adjust",
      method: "post",
      data,
    });
  }

  /** 管理员审核退款申请 */
  static auditRefund(refundId: number, data: BillingRefundAuditForm) {
    return request<BillingRefundVO>({
      url: `/api/v1/ai-billing/refunds/${refundId}/audit`,
      method: "post",
      data,
    });
  }

  /** 退款申请列表（管理端审核中心，需 ai:billing:refund） */
  static getRefunds(query?: BillingRefundQuery) {
    return request<PageResult<BillingRefundVO[]>>({
      url: "/api/v1/ai-billing/refunds",
      method: "get",
      params: query,
    });
  }

  // ==================== 异常监控（管理端） ====================

  /** 异常计费监控列表（需 ai:billing:stat） */
  static getAnomalies(query?: AnomalyRecordQuery) {
    return request<PageResult<AnomalyRecordVO[]>>({
      url: "/api/v1/ai-billing/anomalies",
      method: "get",
      params: query,
    });
  }

  // ==================== 成本管理（管理端） ====================

  /** 模型成本配置列表 */
  static getCosts(query?: ModelCostQuery) {
    return request<PageResult<ModelCostVO[]>>({
      url: "/api/v1/ai-billing/costs",
      method: "get",
      params: query,
    });
  }

  /** 新增模型成本配置 */
  static createCost(data: ModelCostForm) {
    return request<ModelCostVO>({
      url: "/api/v1/ai-billing/costs",
      method: "post",
      data,
    });
  }

  /** 更新模型成本配置 */
  static updateCost(id: number, data: Partial<ModelCostForm>) {
    return request<ModelCostVO>({
      url: `/api/v1/ai-billing/costs/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除模型成本配置 */
  static deleteCost(id: number) {
    return request({
      url: `/api/v1/ai-billing/costs/${id}`,
      method: "delete",
    });
  }

  /** 成本统计（按成本类型聚合） */
  static getCostStats(query?: CostStatQuery) {
    return request<CostStatVO[]>({
      url: "/api/v1/ai-billing/cost-stats",
      method: "get",
      params: query,
    });
  }

  // ==================== 对账 ====================

  /** 对账数据导入（POST /ai-billing/reconcile/import） */
  static importReconcile(data: ImportReconcileForm) {
    return request<{ imported: number }>({
      url: "/api/v1/ai-billing/reconcile/import",
      method: "post",
      data,
    });
  }
}

export default AiBillingAPI;
