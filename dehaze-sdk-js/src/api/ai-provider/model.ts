import { EnabledStatus, PageQuery } from "@/types";

// ==================== AI 模型供应商管理 ====================

/** 供应商健康状态：healthy-健康，suspicious-可疑，open-熔断 */
export type ProviderHealth = "healthy" | "suspicious" | "open";

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
