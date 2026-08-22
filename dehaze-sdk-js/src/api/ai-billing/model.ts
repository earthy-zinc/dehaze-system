import { PageQuery } from "@/types";

// ==================== 枚举类型 ====================

/** 计费记录类型 */
export type BillingType = "chat" | "tool_llm" | "kb_inject" | "asr" | "tts";

/** 流水来源 */
export type CreditLogSource =
  "recharge" | "vip_gift" | "trial" | "admin_adjust" | "refund" | "consume" | "vip_gift_expire";

/** AI 计费退款状态 */
export type BillingRefundStatus = "pending" | "approved" | "rejected";

/** 统计分组维度 */
export type BillingStatGroupBy = "user" | "model" | "billType" | "day";

// ==================== 余额与配额 ====================

/** 用户余额与配额使用概况 */
export interface BalanceVO {
  /** 用户 ID */
  userId: number;
  /** 积分余额 */
  creditsBalance: number;
  /** 是否欠费 */
  arrearsStatus: boolean;
  /** 今日已用积分 */
  dailyUsed: number;
  /** 日积分限额 */
  dailyLimit: number;
  /** 本月已用积分 */
  monthlyUsed: number;
  /** 月积分限额 */
  monthlyLimit: number;
}

// ==================== 计费明细 ====================

/** 计费明细查询参数 */
export interface BillingRecordQuery extends PageQuery {
  conversationId?: number;
  /** 计费类型筛选 */
  billType?: BillingType;
  modelId?: string;
  dateStart?: string;
  dateEnd?: string;
}

/** 计费明细记录 */
export interface BillingRecordVO {
  id: number;
  userId: number;
  conversationId?: number;
  messageId?: number;
  /** 用户选择的模型 */
  model: string;
  /** 实际使用的模型（降级后可能不同） */
  actualModel?: string;
  billType: BillingType;
  inputTokens: number;
  /** 缓存命中的输入 Token */
  cachedInputTokens: number;
  outputTokens: number;
  /** 本次计费积分消耗 */
  credits: number;
  /** 缓存节省的积分 */
  creditsSaved: number;
  /** 工具推理 LLM Token 积分（bill_type=tool_llm 时记录） */
  toolCredits?: number;
  /** 配额消耗积分 */
  quotaConsumed: number;
  /** 预扣积分 */
  preDeduct: number;
  createTime: string;
}

// ==================== 流水 ====================

/** 余额变动流水查询参数 */
export interface CreditLogQuery extends PageQuery {
  source?: CreditLogSource;
  dateStart?: string;
  dateEnd?: string;
}

/** 余额变动流水 */
export interface CreditLogVO {
  id: number;
  userId: number;
  source: CreditLogSource;
  /** 变动金额（正为增加，负为扣减） */
  amount: number;
  /** 变动后余额 */
  balanceAfter: number;
  /** 关联业务 ID（如计费记录 ID、退款 ID） */
  relatedId?: number;
  reason: string;
  operatorId?: number;
  createTime: string;
}

// ==================== 账单 ====================

/** 月结账单 */
export interface BillVO {
  userId: number;
  /** 账单月份，格式 yyyy-MM */
  month: string;
  /** 总消耗积分 */
  totalConsume: number;
  /** 总充值积分 */
  totalRecharge: number;
  /** 总退款积分 */
  totalRefund: number;
  /** 月初余额 */
  balanceStart: number;
  /** 月末余额 */
  balanceEnd: number;
  /** 按 bill_type 维度细分的消耗汇总 */
  itemSummary: Record<BillingType, number>;
}

// ==================== 退款 ====================

/** AI 计费退款申请表单 */
export interface BillingRefundApplyForm {
  /** 原计费记录 ID */
  billingId: number;
  amount: number;
  reason: string;
}

/** AI 计费退款记录 */
export interface BillingRefundVO {
  id: number;
  userId: number;
  billingId: number;
  amount: number;
  reason: string;
  status: BillingRefundStatus;
  /** 审核人 */
  auditorId?: number;
  auditRemark?: string;
  createTime: string;
  updateTime?: string;
}

/** AI 计费退款审核表单 */
export interface BillingRefundAuditForm {
  /** 审核结果：true 通过，false 驳回 */
  approved: boolean;
  auditRemark?: string;
}

// ==================== 管理员操作 ====================

/** 管理员统计查询参数 */
export interface BillingStatQuery {
  userId?: number;
  modelId?: string;
  billType?: BillingType;
  dateStart?: string;
  dateEnd?: string;
  groupBy: BillingStatGroupBy;
}

/** 统计聚合结果 */
export interface BillingStatVO {
  /** 分组维度值 */
  dimension: string;
  totalCredits: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  /** 缓存命中率 */
  cacheHitRate: number;
  /** 缓存节省积分 */
  creditsSaved: number;
  /** 降级次数 */
  degradationCount: number;
}

/** 管理员手动调整积分表单 */
export interface CreditAdjustForm {
  userId: number;
  /** 调整金额（正为增加，负为扣减） */
  amount: number;
  reason: string;
}
