import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  AutoRenewConfigForm,
  AutoRenewConfigVO,
  BalanceRefundForm,
  MyOrderQuery,
  MyOrderVO,
  OrderCreateForm,
  OrderDetailVO,
  OrderPageVO,
  OrderQuery,
  OrderStatsVO,
  PayRequest,
  PayResult,
  RefundApplyForm,
  RefundAuditForm,
  RefundQuery,
  RefundRecordVO,
} from "./model";

class OrderAPI {
  /** 用户端：创建订单 */
  static create(data: OrderCreateForm) {
    return request<PayResult>({
      url: "/api/v1/orders",
      method: "post",
      data,
    });
  }

  /** 用户端：我的订单列表 */
  static listMy(queryParams: MyOrderQuery) {
    return request<PageResult<MyOrderVO[]>>({
      url: "/api/v1/orders/my",
      method: "get",
      params: queryParams,
    });
  }

  /** 用户端/后台：订单详情 */
  static getDetail(orderNo: string) {
    return request<OrderDetailVO>({
      url: "/api/v1/orders/" + orderNo,
      method: "get",
    });
  }

  /** 用户端：取消订单 */
  static cancel(orderNo: string, reason: string) {
    return request({
      url: `/api/v1/orders/${orderNo}/cancel`,
      method: "put",
      params: { reason },
    });
  }

  /** 用户端：发起支付 */
  static pay(orderNo: string, data: PayRequest) {
    return request<PayResult>({
      url: `/api/v1/orders/${orderNo}/pay`,
      method: "post",
      data,
    });
  }

  /** 用户端：申请退款 */
  static applyRefund(orderNo: string, data: RefundApplyForm) {
    return request({
      url: `/api/v1/orders/${orderNo}/refund`,
      method: "post",
      data,
    });
  }

  /** 用户端：余额退款 */
  static balanceRefund(data: BalanceRefundForm) {
    return request({
      url: "/api/v1/orders/balance-refund",
      method: "post",
      data,
    });
  }

  /** 用户端：修改自动续费设置 */
  static updateAutoRenewConfig(data: AutoRenewConfigForm) {
    return request({
      url: "/api/v1/orders/auto-renew/config",
      method: "put",
      data,
    });
  }

  /** 用户端：查询自动续费配置 */
  static getAutoRenewConfig(packageId: number) {
    return request<AutoRenewConfigVO>({
      url: "/api/v1/orders/auto-renew/config",
      method: "get",
      params: { packageId },
    });
  }

  /** 后台：订单分页列表 */
  static getPage(queryParams: OrderQuery) {
    return request<PageResult<OrderPageVO[]>>({
      url: "/api/v1/orders/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：退款审核列表 */
  static listRefunds(queryParams: RefundQuery) {
    return request<PageResult<RefundRecordVO[]>>({
      url: "/api/v1/orders/refunds/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：退款审核通过 */
  static approveRefund(refundId: number, data: RefundAuditForm) {
    return request({
      url: `/api/v1/orders/refunds/${refundId}/approve`,
      method: "put",
      data,
    });
  }

  /** 后台：退款审核驳回 */
  static rejectRefund(refundId: number, data: RefundAuditForm) {
    return request({
      url: `/api/v1/orders/refunds/${refundId}/reject`,
      method: "put",
      data,
    });
  }

  /** 后台：订单统计 */
  static getStats(startTime?: string, endTime?: string) {
    return request<OrderStatsVO>({
      url: "/api/v1/orders/stats",
      method: "get",
      params: { startTime, endTime },
    });
  }
}

export default OrderAPI;
