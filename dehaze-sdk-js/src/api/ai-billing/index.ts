import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  BalanceVO,
  BillVO,
  BillingRecordQuery,
  BillingRecordVO,
  BillingStatQuery,
  BillingStatVO,
  BillingRefundApplyForm,
  BillingRefundAuditForm,
  BillingRefundVO,
  CreditAdjustForm,
  CreditLogQuery,
  CreditLogVO,
} from "./model";

/**
 * AI 计费管理 API
 *
 * 余额查询、计费明细、流水、账单、退款申请（用户端）；
 * 统计、手动调整、退款审核（管理员端）。
 */
class AiBillingAPI {
  // ==================== 用户端接口 ====================

  /** 查询当前用户余额与配额使用情况 */
  static getBalance() {
    return request<BalanceVO>({
      url: "/api/v1/ai-billing/balance",
      method: "get",
    });
  }

  /** 分页查询当前用户计费明细 */
  static getRecords(query?: BillingRecordQuery) {
    return request<PageResult<BillingRecordVO[]>>({
      url: "/api/v1/ai-billing/records",
      method: "get",
      params: query,
    });
  }

  /** 分页查询当前用户余额变动流水 */
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
}

export default AiBillingAPI;
