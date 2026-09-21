import { PageQuery } from "@/types";

// ==================== 枚举类型 ====================

/** 计费记录类型 */
export type BillingType = "chat" | "tool_llm" | "kb_inject" | "asr" | "tts";

/** 流水来源 */
export type CreditLogSource =
  "recharge" | "vip_gift" | "trial" | "admin_adjust" | "refund" | "consume" | "vip_gift_expire";

/** AI 计费退款状态（与后端 sys_ai_refund.status tinyint 一致：1待审核/2已通过/3已驳回） */
export type BillingRefundStatus = 1 | 2 | 3;

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
  /** 指定用户，管理端下钻查看；不传为当前登录用户 */
  userId?: number;
}

/** 计费明细记录 */
export interface BillingRecordVO {
  id: number;
  userId: number;
  conversationId?: number;
  messageId?: number;
  /** 实际使用的模型（降级场景为降级模型） */
  model: string;
  /** 用户原选模型（降级时记录原模型，未降级为空） */
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
  /** 指定用户，管理端下钻查看；不传为当前登录用户 */
  userId?: number;
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

/** 退款申请列表查询参数（管理端审核中心） */
export interface BillingRefundQuery extends PageQuery {
  /** 退款状态筛选：1待审核 / 2已通过 / 3已驳回 */
  status?: number;
  /** 用户 ID 筛选 */
  userId?: number;
  dateStart?: string;
  dateEnd?: string;
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
  /**
   * 异常规则类型：single_high-单次超高 / burst-突发峰值 /
   * consecutive_quota_fail-连续配额不足 / empty_high_output-空回复高耗
   */
  anomalyType?: string;
  /** 处理状态：0-待处理，1-已处理，2-已忽略 */
  status?: 0 | 1 | 2;
  /** 指定用户筛选（对齐后端实际行为：app/router/ai_billing.py list_anomalies 支持 userId 查询参数） */
  userId?: number;
  dateStart?: string;
  dateEnd?: string;
}

/** 异常计费记录 */
export interface AnomalyRecordVO {
  id: number;
  userId: number;
  /** 关联计费记录 ID（配额不足类异常无关联记录） */
  billingId?: number;
  /** 异常规则类型，取值同 AnomalyRecordQuery.anomalyType */
  anomalyType: string;
  /** 异常详情描述 */
  detail: string;
  /** 处理状态：0-待处理，1-已处理，2-已忽略 */
  status: 0 | 1 | 2;
  /** 触发时间 */
  triggerAt: string;
  createTime?: string;
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

/** 模型成本配置新增表单（价格版本主表 + 档位明细，与用户售价表结构对称） */
export interface ModelCostForm {
  modelId: string;
  /** 供应商 ID（对齐后端实际行为：app/models/schema/ai_billing_cost.py ModelCostCreateRequest.provider_id 为必填） */
  providerId: number;
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

/** 模型成本配置更新表单（仅版本主表字段，档位明细随新增版本生成，不可单独更新） */
export interface ModelCostUpdateForm {
  /** 币种 */
  currency?: string;
  /** 生效时间 */
  effectiveFrom?: string;
  /** 失效时间 */
  effectiveTo?: string;
  /** 状态：1-启用，0-停用 */
  status?: 0 | 1;
}

/** 模型成本配置（价格版本） */
export interface ModelCostVO extends ModelCostForm {
  id: number;
  /** 价格版本号（同模型同供应商内递增） */
  priceVersion: number;
  createTime?: string;
  updateTime?: string;
}

/** 成本统计查询参数（groupBy=overall 时为整体双口径毛利；model/provider 为成本分组分解） */
export interface CostStatQuery {
  startTime?: string;
  endTime?: string;
  /** 分组维度：overall-整体双口径（默认）/ model-按模型 / provider-按供应商 */
  groupBy?: "overall" | "model" | "provider";
  modelId?: string;
  providerId?: number;
}

/**
 * 成本-利润统计项（按模型/供应商/时间，收入/成本/毛利双口径）。
 * 整体毛利为官方口径；AI 参考毛利为辅助口径。
 */
export interface CostStatVO {
  /** 统计维度值 */
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

/**
 * 成本分组统计项（groupBy=model/provider）：订单实收无法按模型/供应商归因，
 * 仅返回维度值与成本分解，不携带收入/毛利/口径。
 */
export interface CostGroupVO {
  /** 维度值：模型标识 / 供应商 ID */
  dimension: string;
  /** 成本（Σ sys_ai_billing.cost） */
  cost: number;
}

// ==================== 对账 ====================

/** 对账数据导入表单 */
export interface ImportReconcileForm {
  /** 对账数据内容 */
  content: string;
  /** 对账周期起（对齐后端实际行为：app/models/schema/ai_billing_cost.py ReconcileImportRequest.start_time 可空） */
  startTime?: string;
  /** 对账周期止（同上，可空） */
  endTime?: string;
}
