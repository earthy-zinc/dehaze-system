import type {
  OrderStatus,
  PayMethod,
  RefundStatus,
  RefundReasonType,
} from "dehaze-sdk-js";
import type { TagType } from "@/enums/TagType";

interface EnumOption<T> {
  label: string;
  value: T;
  tag: TagType;
}

export const orderStatusOptions: EnumOption<OrderStatus>[] = [
  { label: "待支付", value: "pending", tag: "warning" },
  { label: "已支付", value: "paid", tag: "primary" },
  { label: "已完成", value: "completed", tag: "info" },
  { label: "已取消", value: "cancelled", tag: "info" },
  { label: "退款中", value: "refunding", tag: "warning" },
  { label: "已退款", value: "refunded", tag: "info" },
];

export const payMethodOptions: { label: string; value: PayMethod }[] = [
  { label: "微信支付", value: "wechat" },
  { label: "支付宝", value: "alipay" },
  { label: "余额支付", value: "balance" },
  { label: "组合支付", value: "combined" },
];

export const refundStatusOptions: EnumOption<RefundStatus>[] = [
  { label: "退款中", value: "refunding", tag: "warning" },
  { label: "退款成功", value: "refunded", tag: "info" },
  { label: "退款失败", value: "refund_failed", tag: "danger" },
];

export const refundReasonOptions: { label: string; value: RefundReasonType }[] =
  [
    { label: "售后问题", value: "after_sale" },
    { label: "不可抗原因", value: "force_majeure" },
    { label: "商家原因", value: "merchant" },
    { label: "其他原因", value: "other" },
  ];
