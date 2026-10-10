import { EnabledStatus, PageQuery } from "@/types";

// ==================== AI 模型管理 ====================

/**
 * 模型类型（全模态统一白名单）：
 * chat-对话、embedding-向量、rerank-重排、image_gen-文生图、image_edit-图像编辑（去雾等）、
 * video_gen-视频生成、asr-语音识别、tts-语音合成、music_gen-音乐生成、3d_gen-3D 生成。
 */
export type AiModelType =
  | "chat"
  | "embedding"
  | "rerank"
  | "image_gen"
  | "image_edit"
  | "video_gen"
  | "asr"
  | "tts"
  | "music_gen"
  | "3d_gen";

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
  /**
   * 任务归类标签（如图像模型 ["dehaze"]/["super_resolution"]、asr ["stream"]/["offline"]、
   * embedding ["text"]/["image"]）；按 model_type + task_tags 过滤选择场景
   */
  taskTags?: string[] | null;
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
  /** 按模型类型筛选 */
  modelType?: AiModelType;
  /** 按任务标签筛选（逗号分隔，OR 语义：task_tags 数组含任一即命中） */
  taskTags?: string | null;
}

/** 模型视图对象 */
export interface AiModelVO {
  id: number;
  providerId: number;
  /** 模型业务标识 */
  modelId: string;
  modelType: AiModelType;
  /** 任务归类标签（图/语音/向量模型用于选择场景过滤） */
  taskTags?: string[] | null;
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
  /**
   * 是否其模型类型的平台默认模型（后端 `sys_dict: ai_model.default_model[.{modelType}]`）。
   *
   * 仅 `listEnabledModels`（GET /ai/models/enabled）计算并返回：命中平台默认的模型为 true，
   * 其余为 false；管理端分页列表不返回（键缺失）。图像页模型预选＝保留既定的"历史最近使用"
   * 优先，**无历史时以此标记为准**；后端未配置该类型默认时全部为 false（不下选），
   * 不得回落到"启用列表首项"猜值兜底。
   */
  isDefault?: boolean | null;
  /** 最近可用性测试结果：0-未测试，1-可用，2-不可用 */
  lastTestStatus: number;
  /** 最近测试时间 */
  lastTestAt?: string | null;
  /** 最近测试失败原因 */
  lastTestError?: string | null;
  /** 近 24h 真实调用次数（null=未调用） */
  calls24h?: number | null;
  /** 近 24h 成功率（百分比） */
  successRate24h?: number | null;
  /** 最近一次真实调用时间 */
  lastCallAt?: string | null;
  createTime?: string | null;
}

// ==================== 模型用户售价（价格版本化） ====================

/**
 * 价格档位 token 类型：input-输入、cached-缓存命中、output-输出；
 * unit-非 token 单档（仅 usage_kind='unit' 的价格版本使用，不设峰谷/分段，
 * 与主表 billing_unit 联动，值域对齐后端 token_type 列）
 */
export type ModelPriceTokenType = "input" | "cached" | "output" | "unit";

/** 价格时段档位：peak-高峰、idle-闲时 */
export type ModelPriceTimeSlot = "peak" | "idle";

/**
 * 计价单位（价格版本维度化，决定 detail 档位结构）：
 * million_tokens-百万 token（chat/embedding/rerank）、image-张（image_gen/image_edit）、
 * second-秒（asr）、ten_thousand_chars-万字符（tts）、frame-帧（video_gen 预留）
 */
export type ModelPriceBillingUnit =
  "million_tokens" | "image" | "second" | "ten_thousand_chars" | "frame";

/** 档位计量口径：token-按 token 三维档位（token_type×time_slot×分段）；unit-非 token 单档 */
export type ModelPriceUsageKind = "token" | "unit";

/** 价格档位明细表单 */
export interface ModelPriceDetailForm {
  /** 计量口径（默认 token；非 token 模型填 unit，单档不设峰谷与分段） */
  usageKind?: ModelPriceUsageKind;
  tokenType: ModelPriceTokenType;
  timeSlot: ModelPriceTimeSlot;
  /** 上下文分段下界（按 输入+cached 总量匹配） */
  minTokens?: number;
  /** 上下文分段上界（空表示不限） */
  maxTokens?: number | null;
  /** 单价（积分/计价单位），提交数字即可 */
  unitPrice: number;
}

/** 价格版本创建表单（同模型同供应商生成新版本，历史版本保留） */
export interface ModelPriceForm {
  /** 模型标识（业务键，含 `/` 的厂商模型名随请求体提交） */
  modelId: string;
  /** 供应商 ID */
  providerId: number;
  /** 计价单位（默认 million_tokens；决定 detail 档位结构） */
  billingUnit?: ModelPriceBillingUnit;
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
  /** 计量口径（token-三维档位；unit-非 token 单档） */
  usageKind: ModelPriceUsageKind;
  tokenType: ModelPriceTokenType;
  timeSlot: ModelPriceTimeSlot;
  minTokens: number;
  maxTokens?: number | null;
  /** 单价（积分/计价单位）。后端 Decimal 序列化为字符串，计算前需自行转 Number */
  unitPrice: string;
}

/** 价格版本视图对象 */
export interface ModelPriceVO {
  id: number;
  modelId: string;
  providerId: number;
  /** 价格版本号（同模型同供应商递增） */
  priceVersion: number;
  /** 计价单位（维度化：百万token/张/秒/万字符/帧） */
  billingUnit: ModelPriceBillingUnit;
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

// ==================== 用户偏好默认模型 ====================

/**
 * 用户偏好默认模型（未设置时 `defaultModel` 键不出现——响应统一剔除 null，
 * 与全局 NON_NULL 序列化口径一致，消费方按缺失/null 等价处理：新建会话回落平台默认模型）
 */
export interface AiUserPreference {
  defaultModel?: string | null;
}

/** 用户偏好默认模型表单（defaultModel 为 null 表示清除偏好） */
export interface AiUserPreferenceForm {
  defaultModel: string | null;
}
