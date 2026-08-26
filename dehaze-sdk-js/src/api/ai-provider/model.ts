import { EnabledStatus, PageQuery } from "@/types";

// ==================== AI 模型供应商管理 ====================

/** 供应商健康状态：healthy-健康，suspicious-可疑，open-熔断 */
export type ProviderHealth = "healthy" | "suspicious" | "open";

/**
 * 用户身份透传配置（user_identity_forward）。
 * 抽象覆盖不同厂商的用户标识差异：DeepSeek user_id / OpenAI user / Anthropic metadata.user_id。
 */
export interface UserIdentityForwardConfig {
  /** 是否启用透传 */
  enabled: boolean;
  /** 透传字段名（user_id / user / metadata.user_id 等） */
  field: string;
  /** 前缀 */
  prefix?: string;
  /** 最大长度 */
  maxLen?: number;
}

/** 创建供应商表单 */
export interface ProviderCreateForm {
  /** 供应商编码 */
  providerCode: string;
  /** 显示名称 */
  displayName: string;
  /** API 基础地址 */
  apiBaseUrl: string;
  /** 协议类型（默认 openai_compat） */
  protocolType?: string;
  /** 认证方式（默认 bearer） */
  authType?: string;
  /** 默认请求头（JSON） */
  defaultHeaders?: Record<string, unknown> | null;
  /** 排序序号 */
  sortOrder?: number;
  /** 健康检查开关：1-开启（参与熔断判定），0-关闭 */
  healthCheckEnabled?: EnabledStatus;
  /** 用户身份透传配置 */
  userIdentityForward?: UserIdentityForwardConfig | null;
  /** 运维备注 */
  remark?: string | null;
  /** 状态：1-启用，0-禁用 */
  status?: EnabledStatus;
}

/** 更新供应商表单 */
export interface ProviderUpdateForm {
  displayName?: string;
  apiBaseUrl?: string;
  protocolType?: string;
  authType?: string;
  defaultHeaders?: Record<string, unknown> | null;
  sortOrder?: number;
  healthCheckEnabled?: EnabledStatus;
  /** 用户身份透传配置 */
  userIdentityForward?: UserIdentityForwardConfig | null;
  remark?: string | null;
  status?: EnabledStatus;
}

/** 供应商视图对象 */
export interface ProviderVO {
  id: number;
  /** 供应商编码 */
  providerCode: string;
  displayName: string;
  apiBaseUrl: string;
  protocolType: string;
  authType: string;
  defaultHeaders?: Record<string, unknown> | null;
  sortOrder: number;
  healthCheckEnabled: EnabledStatus;
  /** 用户身份透传配置 */
  userIdentityForward?: UserIdentityForwardConfig | null;
  remark?: string | null;
  /** 健康状态：healthy/suspicious/open */
  health?: ProviderHealth | null;
  status: EnabledStatus;
  createTime?: string | null;
  updateTime?: string | null;
}

/** 供应商分页查询参数 */
export interface ProviderPageQuery extends PageQuery {
  /** 关键字（按显示名称/供应商编码模糊搜索） */
  keyword?: string;
}

// ==================== API Key 管理 ====================

/** 创建 API Key 表单 */
export interface ProviderKeyCreateForm {
  /** Key 名称 */
  name: string;
  /** Key 明文（服务端加密后存储，响应不含明文） */
  key: string;
  /** 优先级（数字越小越优先） */
  priority?: number;
  /** 权重 */
  weight?: number;
  /** 日调用上限 */
  dailyQuota?: number | null;
  /** Key 级 RPM 限流 */
  rpmLimit?: number | null;
  /** 过期时间 */
  expiresAt?: string | null;
  /** 状态：1-启用，0-禁用 */
  status?: EnabledStatus;
}

/** 更新 API Key 表单 */
export interface ProviderKeyUpdateForm {
  name?: string;
  priority?: number;
  weight?: number;
  status?: EnabledStatus;
  dailyQuota?: number | null;
  /** Key 级 RPM 限流 */
  rpmLimit?: number | null;
  expiresAt?: string | null;
}

/**
 * API Key 视图对象。
 * 注意：创建/查询响应均不含明文，仅返回密钥前缀等展示字段。
 */
export interface ProviderKeyVO {
  id: number;
  /** 关联供应商 ID */
  providerId: number;
  name: string;
  /** 密钥前缀（展示用，非明文） */
  keyPrefix?: string | null;
  status: EnabledStatus;
  priority: number;
  weight: number;
  dailyQuota?: number | null;
  /** Key 级 RPM 限流 */
  rpmLimit?: number | null;
  expiresAt?: string | null;
  /** 最后使用时间 */
  lastUsedAt?: string | null;
  /** 最后使用的用户 ID */
  lastUsedBy?: number | null;
  createTime?: string | null;
  updateTime?: string | null;
}

/** 供应商连通性测试结果（后台执行，结构因供应商而异） */
export type ConnectionTestResult = Record<string, unknown>;

// ==================== 运营统计（管理端） ====================

/** 运营统计查询参数（需 ai:model:manage） */
export interface UsageStatQuery {
  startTime?: string;
  endTime?: string;
  /** 聚合粒度：hour-小时 / day-天 */
  granularity?: "hour" | "day";
}

/** 供应商健康看板项（成功率/429/P95/熔断） */
export interface ProviderHealthStatVO {
  providerId: number;
  providerName: string;
  /** 健康状态：healthy/suspicious/open */
  health: ProviderHealth;
  callCount: number;
  /** 成功率（0-100） */
  successRate: number;
  /** HTTP 429 次数 */
  rate429: number;
  /** 延迟 P95（毫秒） */
  p95LatencyMs?: number;
  /** 是否熔断 */
  circuitOpen: boolean;
}

/** 模型用量分布项（Token/调用数/积分开销） */
export interface ModelUsageStatVO {
  modelId: string;
  displayName: string;
  callCount: number;
  inputTokens: number;
  outputTokens: number;
  /** 积分开销 */
  credits: number;
}

/** 降级频率项 */
export interface DowngradeStatVO {
  modelId: string;
  /** 降级次数 */
  count: number;
}

/** 降级与故障统计 */
export interface DegradeFaultStatVO {
  /** 各模型降级频率 */
  downgradeFrequency: DowngradeStatVO[];
  /** Key 失败切换次数 */
  keyFailoverCount: number;
  /** 故障次数 */
  faultCount: number;
}

/** 运营统计视图对象（GET /api/v1/ai/usage/stats） */
export interface UsageStatsVO {
  providerHealth: ProviderHealthStatVO[];
  modelUsage: ModelUsageStatVO[];
  degradeFault: DegradeFaultStatVO;
}
