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
  /** 误扣申诉状态：0-无，1-待审核，2-已通过，3-已驳回 */
  refundStatus?: 0 | 1 | 2 | 3;
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

// ==================== 消耗汇总（用户端） ====================

/** 消耗趋势点（日/月维度） */
export interface BillingTrendPointVO {
  date: string;
  credits: number;
  inputTokens: number;
  outputTokens: number;
}

/** 模型消耗分布项 */
export interface BillingModelDistVO {
  model: string;
  credits: number;
  tokens: number;
}

/** 缓存节省汇总 */
export interface BillingSavingsVO {
  cachedInputTokens: number;
  creditsSaved: number;
}

/** 当前时段消耗汇总（GET /ai-billing/summary，含日/月趋势、模型分布、节省汇总） */
export interface BillingSummaryVO {
  /** 当前时段总消耗积分 */
  totalCredits: number;
  /** 当前时段输入Token总数 */
  inputTokens: number;
  /** 当前时段输出Token总数 */
  outputTokens: number;
  /** 日/月消耗趋势 */
  trend?: BillingTrendPointVO[];
  /** 模型分布 */
  modelDistribution?: BillingModelDistVO[];
  /** 缓存节省汇总 */
  savings?: BillingSavingsVO;
}

// ==================== 异常监控（管理端） ====================

/** 异常计费记录查询参数（需 ai:billing:stat） */
export interface AnomalyRecordQuery extends PageQuery {
  /** 异常类型：anomalous-异常计费 / manual-人工调整 / auto_compensated-自动补偿 */
  anomalyType?: string;
  dateStart?: string;
  dateEnd?: string;
}

/** 异常计费记录 */
export interface AnomalyRecordVO {
  id: number;
  userId: number;
  username?: string;
  billingId?: number;
  anomalyType: string;
  /** 双口径：理论成本与实收积分 */
  costCredits?: number;
  credits?: number;
  reason?: string;
  /** 处理状态：pending/compensated/ignored */
  status?: string;
  createTime: string;
}

// ==================== 成本管理（管理端） ====================

/** 模型成本配置查询参数（按模型/供应商/版本） */
export interface ModelCostQuery extends PageQuery {
  keyword?: string;
  modelId?: string;
  providerId?: number;
}

/** 成本单价档位明细（token 类型 × 上下文分段 × 时段，元/百万 token） */
export interface ModelCostDetailForm {
  /** token 类型：input/cached/output */
  tokenType: "input" | "cached" | "output";
  /** 时段：peak-高峰 / idle-空闲 */
  timeSlot: "peak" | "idle";
  /** 上下文分段下限（0 表示不限制） */
  minTokens?: number;
  /** 上下文分段上限（NULL 表示不限制） */
  maxTokens?: number;
  /** 单位价格（元/百万 token） */
  unitPrice: number;
}

/** 模型成本配置表单（价格版本主表 + 档位明细，与用户售价表结构对称） */
export interface ModelCostForm {
  modelId: string;
  /** 供应商 ID */
  providerId?: number;
  /** 币种（默认 CNY） */
  currency?: string;
  /** 生效时间 */
  effectiveFrom?: string;
  /** 失效时间 */
  effectiveTo?: string;
  /** 状态：1-启用，0-停用 */
  status?: 0 | 1;
  /** 档位明细 */
  details?: ModelCostDetailForm[];
}

/** 模型成本配置（价格版本） */
export interface ModelCostVO extends ModelCostForm {
  id: number;
  /** 价格版本号（同模型同供应商内递增） */
  priceVersion: number;
  createTime?: string;
  updateTime?: string;
}

/** 成本统计查询参数 */
export interface CostStatQuery {
  startTime?: string;
  endTime?: string;
  modelId?: string;
  providerId?: number;
}

/**
 * 成本-利润统计项（按模型/供应商/时间，收入/成本/毛利双口径）。
 * 整体毛利为官方口径；AI 参考毛利为辅助口径。
 */
export interface CostStatVO {
  /** 统计维度值（model/provider/时间） */
  dimension?: string;
  /** 收入（订单实收） */
  revenue: number;
  /** 成本（模型调用估算成本 Σ sys_ai_billing.cost） */
  cost: number;
  /** 毛利（收入 − 成本） */
  profit: number;
  /** 毛利率 */
  profitRate: number;
  /** 口径：overall-整体官方 / ai-参考毛利 */
  metric: "overall" | "ai";
}

// ==================== 对账 ====================

/** 对账数据导入表单 */
export interface ImportReconcileForm {
  /** 对账数据内容 */
  content: string;
  /** 对账周期起 */
  startTime: string;
  /** 对账周期止 */
  endTime: string;
}
