import type {
  FeedbackStatus,
  FeedbackType,
  FeedbackReplyType,
} from "dehaze-sdk-js";
import type { TagType } from "@/enums/TagType";

interface EnumOption<T> {
  label: string;
  value: T;
  tag: TagType;
}

export const feedbackTypeOptions: EnumOption<FeedbackType>[] = [
  { label: "功能建议", value: "suggestion", tag: "primary" },
  { label: "问题报告", value: "bug", tag: "danger" },
  { label: "体验反馈", value: "experience", tag: "success" },
  { label: "投诉", value: "complaint", tag: "warning" },
];

export const feedbackStatusOptions: EnumOption<FeedbackStatus>[] = [
  { label: "待处理", value: "pending", tag: "warning" },
  { label: "处理中", value: "processing", tag: "primary" },
  { label: "已回复", value: "replied", tag: "success" },
  { label: "已关闭", value: "closed", tag: "info" },
];

export const feedbackReplyTypeOptions: EnumOption<FeedbackReplyType>[] = [
  { label: "通知", value: "info", tag: "info" },
  { label: "已解决", value: "resolved", tag: "success" },
  { label: "不支持", value: "unsupported", tag: "info" },
  { label: "转开发", value: "dev_transfer", tag: "warning" },
];

export const feedbackModuleOptions: { label: string; value: string }[] = [
  { label: "去雾处理", value: "dehaze" },
  { label: "指标评估", value: "evaluate" },
  { label: "数据集", value: "dataset" },
  { label: "会员", value: "member" },
  { label: "套餐", value: "package" },
  { label: "订单", value: "order" },
  { label: "其他", value: "other" },
];
