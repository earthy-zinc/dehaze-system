import { EnabledStatus, PageQuery } from "@/types";

// ==================== AI 模型管理 ====================

/** 模型类型：chat-对话、embedding-向量、rerank-重排 */
export type AiModelType = "chat" | "embedding" | "rerank";

/** 模型速度档位（由供应商健康快照 P95 延迟推导） */
export type AiModelSpeedTier = "fast" | "medium" | "slow" | "unknown";

/**
 * 模型创建表单。
 *
 * model_type / dimension 创建后不可改（后端更新接口传这两个字段会拒绝），
 * 因此仅出现在创建表单中。
 */
export interface AiModelForm {
  /** 关联供应商 ID */
  providerId: number;
  /** 模型标识（1-64 字符，同一供应商下唯一） */
  modelId: string;
  /** 模型类型（默认 chat） */
  modelType?: AiModelType;
  /** embedding 向量维度（model_type=embedding 时必填） */
  dimension?: number;
  /** 显示名称 */
  displayName: string;
  /** 最大上下文 Token 数（默认 4096） */
  maxContextTokens?: number;
  /** 最大输出 Token 数（默认 4096） */
  maxOutputTokens?: number;
  /** 是否支持多模态 */
  supportsMultimodal?: boolean;
  /** 是否支持工具调用 */
  supportsToolCall?: boolean;
  /** 是否支持流式输出（默认 true） */
  supportsStreaming?: boolean;
  /** 是否支持 Prompt 缓存 */
  supportsPromptCache?: boolean;
  /** 是否支持结构化输出 */
  supportsStructuredOutput?: boolean;
  /** 厂商私有请求参数（如 enable_thinking / reasoning_effort） */
  extraRequestParams?: Record<string, unknown> | null;
  /** 降级模型 ID（关联 sys_ai_model.id） */
  fallbackModelId?: number | null;
  /** Prompt 缓存稳定前缀长度 */
  promptCachePrefixLen?: number;
  /** 状态：1-启用，0-禁用 */
  status?: EnabledStatus;
  /** 最低可用 VIP 等级（0-所有用户；1-VIP1 及以上；2-VIP2 及以上） */
  vipLevel?: number;
}

/** 模型更新表单（全部可选，仅传需要变更的字段） */
export type AiModelUpdateForm = Partial<Omit<AiModelForm, "modelType" | "dimension">>;

/** 模型分页查询参数 */
export interface AiModelQuery extends PageQuery {
  /** 关键字（按显示名称/模型标识模糊搜索） */
  keyword?: string;
  /** 按模型类型筛选（chat/embedding/rerank） */
  modelType?: AiModelType;
}

/** 模型视图对象 */
export interface AiModelVO {
  id: number;
  providerId: number;
  /** 模型业务标识 */
  modelId: string;
  modelType: AiModelType;
  /** embedding 向量维度 */
  dimension?: number | null;
  displayName: string;
  maxContextTokens: number;
  maxOutputTokens: number;
  /** 能力标识（0/1，后端以整型存储） */
  supportsMultimodal: number;
  supportsToolCall: number;
  supportsStreaming: number;
  supportsPromptCache: number;
  supportsStructuredOutput: number;
  /** 厂商私有请求参数 */
  extraRequestParams?: Record<string, unknown> | null;
  /** 降级模型 ID */
  fallbackModelId?: number | null;
  promptCachePrefixLen: number;
  status: EnabledStatus;
  /** 最低可用 VIP 等级 */
  vipLevel: number;
  /** 速度档位 */
  speedTier?: AiModelSpeedTier | null;
  /** 是否作为其他启用模型的降级目标 */
  isFallbackTarget?: boolean | null;
  createTime?: string | null;
}

// ==================== 模型用户售价（价格版本化） ====================

/** 价格档位 token 类型：input-输入、cached-缓存命中、output-输出 */
export type ModelPriceTokenType = "input" | "cached" | "output";

/** 价格时段档位：peak-高峰、idle-闲时 */
export type ModelPriceTimeSlot = "peak" | "idle";

/** 价格档位明细表单 */
export interface ModelPriceDetailForm {
  tokenType: ModelPriceTokenType;
  timeSlot: ModelPriceTimeSlot;
  /** 上下文分段下界（按 输入+cached 总量匹配） */
  minTokens?: number;
  /** 上下文分段上界（空表示不限） */
  maxTokens?: number | null;
  /** 单价（积分/百万 token），提交数字即可 */
  unitPrice: number;
}

/** 价格版本创建表单（同模型同供应商生成新版本，历史版本保留） */
export interface ModelPriceForm {
  /** 模型标识（需与路径参数一致） */
  modelId: string;
  /** 供应商 ID */
  providerId: number;
  /** 单价单位（默认 credits_per_million：积分/百万 token） */
  unit?: string;
  /** 生效时间（为空取当前时间） */
  effectiveFrom?: string;
  /** 失效时间（空表示长期有效） */
  effectiveTo?: string | null;
  /** 状态：1-生效，0-停用 */
  status?: EnabledStatus;
  /** 档位明细 */
  details?: ModelPriceDetailForm[];
}

/** 价格版本更新表单（仅主表字段，档位明细不支持局部更新） */
export interface ModelPriceUpdateForm {
  unit?: string;
  effectiveFrom?: string;
  effectiveTo?: string | null;
  status?: EnabledStatus;
}

/** 价格档位明细视图对象 */
export interface ModelPriceDetailVO {
  id: number;
  /** 关联价格版本 ID */
  priceId: number;
  tokenType: ModelPriceTokenType;
  timeSlot: ModelPriceTimeSlot;
  minTokens: number;
  maxTokens?: number | null;
  /** 单价（积分/百万 token）。后端 Decimal 序列化为字符串，计算前需自行转 Number */
  unitPrice: string;
}

/** 价格版本视图对象 */
export interface ModelPriceVO {
  id: number;
  modelId: string;
  providerId: number;
  /** 价格版本号（同模型同供应商递增） */
  priceVersion: number;
  unit: string;
  effectiveFrom: string;
  effectiveTo?: string | null;
  status: EnabledStatus;
  details: ModelPriceDetailVO[];
  createTime?: string | null;
  updateTime?: string | null;
}

/** 价格版本分页查询参数（page/size，与管理端分页的 pageNum/pageSize 不同） */
export interface ModelPriceQuery {
  page?: number;
  size?: number;
}
