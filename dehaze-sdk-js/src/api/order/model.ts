import { PageQuery } from "@/types";

/** 订单状态(pending:待支付;paid:已支付;completed:已完成;cancelled:已取消;refunding:退款中;refunded:已退款) */
export type OrderStatus = "pending" | "paid" | "completed" | "cancelled" | "refunding" | "refunded";

/** 支付方式 */
export type PayMethod = "wechat" | "alipay" | "balance" | "combined";

/** 退款状态(refunding:退款中;refunded:退款成功;refund_failed:退款失败) */
export type RefundStatus = "refunding" | "refunded" | "refund_failed";

/** 商品类型(vip:会员卡;credit:积分卡) */
export type OrderPackageType = "vip" | "credit";

/** 订单创建表单 */
export interface OrderCreateForm {
  packageId: number;
  couponId?: number;
  payMethod: PayMethod;
  balanceAmount?: number;
}

/** 余额退款表单 */
export interface BalanceRefundForm {
  orderId: number;
  amount: number;
}

/** 订单查询参数（用户端） */
export interface MyOrderQuery extends PageQuery {
  status?: OrderStatus;
}

/** 订单查询参数（后台） */
export interface OrderQuery extends PageQuery {
  orderNo?: string;
  keywords?: string;
  status?: OrderStatus;
  payMethod?: PayMethod;
  amountMin?: number;
  amountMax?: number;
  paidTimeStart?: string;
  paidTimeEnd?: string;
}

/** 支付请求 */
export interface PayRequest {
  payMethod: PayMethod;
}

/** 支付结果 */
export interface PayResult {
  orderNo: string;
  payMethod: PayMethod;
  payUrl?: string;
  qrCode?: string;
  paid: boolean;
}

/** 退款申请表单 */
export interface RefundApplyForm {
  reason: string;
  customReason?: string;
  reasonType?: string;
}

/** 退款审核表单 */
export interface RefundAuditForm {
  approved: boolean;
  remark: string;
}

/** 退款记录查询参数 */
export interface RefundQuery extends PageQuery {
  orderNo?: string;
  keywords?: string;
  status?: RefundStatus;
  applyTimeStart?: string;
  applyTimeEnd?: string;
}

/** 自动续费配置表单 */
export interface AutoRenewConfigForm {
  packageId: number;
  payMethod: PayMethod;
  enabled: boolean;
}

/** 自动续费配置VO */
export interface AutoRenewConfigVO {
  userId: number;
  packageId: number;
  packageName: string;
  payMethod: PayMethod;
  enabled: boolean;
  nextRenewTime?: string;
  failCount: number;
  closeReason?: string;
}

/** 订单列表VO（用户端） */
export interface MyOrderVO {
  id: number;
  orderNo: string;
  packageName: string;
  packageLevel: string;
  packageType: OrderPackageType;
  creditAmount?: number;
  payableAmount: number;
  paidAmount: number;
  payMethod?: PayMethod;
  status: OrderStatus;
  createTime: string;
  paidTime?: string;
  packageExpireTime?: string;
}

/** 订单列表VO（后台） */
export interface OrderPageVO extends MyOrderVO {
  userId: number;
  username: string;
  originalPrice: number;
  discountAmount: number;
  couponAmount: number;
}

/** 支付流水VO */
export interface PaymentRecordVO {
  id: number;
  paymentNo: string;
  channel: PayMethod;
  amount: number;
  status: number;
  callbackTime?: string;
  createTime: string;
}

/** 退款记录VO */
export interface RefundRecordVO {
  id: number;
  refundNo: string;
  orderId: number;
  orderNo: string;
  userId: number;
  username: string;
  refundAmount: number;
  reason: string;
  usedDays?: number;
  usedCredits?: number;
  status: RefundStatus;
  channel?: PayMethod;
  channelRefundNo?: string;
  applyTime: string;
  auditTime?: string;
  auditorId?: number;
  auditRemark?: string;
  refundTime?: string;
  errorMessage?: string;
}

/** 订单详情VO */
export interface OrderDetailVO extends OrderPageVO {
  expireTime: string;
  effectiveTime?: string;
  cancelReason?: string;
  isAutoRenew: number;
  paymentRecords?: PaymentRecordVO[];
  refundRecord?: RefundRecordVO;
}

/** 订单统计VO */
export interface OrderStatsVO {
  totalOrders: number;
  totalRevenue: number;
  totalRefund: number;
  refundRate: number;
  statusDistribution: Record<OrderStatus, number>;
  payMethodDistribution: Record<PayMethod, number>;
  packageDistribution: Array<{
    packageId: number;
    packageName: string;
    count: number;
    revenue: number;
  }>;
  dailyStats: Array<{
    date: string;
    count: number;
    revenue: number;
  }>;
}
