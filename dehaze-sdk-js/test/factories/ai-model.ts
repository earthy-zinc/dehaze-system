import { pageQuery, uniqueCode, uniqueName } from "./common";
import type {
  AiModelForm,
  AiModelQuery,
  AiModelUpdateForm,
  ModelPriceForm,
  ModelPriceQuery,
  ModelPriceUpdateForm,
} from "../../src/api/ai-model/model";

/**
 * 模型创建表单工厂。
 *
 * 对齐后端 AiModelCreate：provider_id 为供应商主键，能力字段平铺为 camelCase；
 * 费率字段已剥离至价格版本（ModelPriceForm），不再挂在模型上。
 */
export const createAiModelForm = (overrides?: Partial<AiModelForm>): AiModelForm => ({
  providerId: 1,
  modelId: uniqueCode("test_model"),
  displayName: uniqueName("测试模型"),
  maxContextTokens: 128000,
  maxOutputTokens: 4096,
  supportsMultimodal: false,
  supportsToolCall: true,
  supportsStreaming: true,
  supportsPromptCache: false,
  supportsStructuredOutput: false,
  promptCachePrefixLen: 0,
  status: 1,
  vipLevel: 0,
  ...overrides,
});

/** 模型更新表单工厂 */
export const createAiModelUpdateForm = (
  overrides?: Partial<AiModelUpdateForm>
): AiModelUpdateForm => ({
  displayName: uniqueName("更新模型"),
  ...overrides,
});

/** 模型分页查询参数工厂 */
export const createAiModelQuery = (overrides?: Partial<AiModelQuery>): AiModelQuery =>
  pageQuery<AiModelQuery>({ ...overrides });

/** 价格版本创建表单工厂（三维档位：token 类型 × 时段 × 上下文分段） */
export const createModelPriceForm = (
  modelId: string,
  providerId: number,
  overrides?: Partial<ModelPriceForm>
): ModelPriceForm => ({
  modelId,
  providerId,
  unit: "credits_per_million",
  status: 1,
  details: [
    { tokenType: "input", timeSlot: "peak", minTokens: 0, unitPrice: 2 },
    { tokenType: "cached", timeSlot: "peak", minTokens: 0, unitPrice: 0.5 },
    { tokenType: "output", timeSlot: "peak", minTokens: 0, unitPrice: 8 },
  ],
  ...overrides,
});

/** 价格版本更新表单工厂 */
export const createModelPriceUpdateForm = (
  overrides?: Partial<ModelPriceUpdateForm>
): ModelPriceUpdateForm => ({
  status: 0,
  ...overrides,
});

/** 价格版本分页查询参数工厂（page/size，非 pageNum/pageSize） */
export const createModelPriceQuery = (overrides?: Partial<ModelPriceQuery>): ModelPriceQuery => ({
  page: 1,
  size: 10,
  ...overrides,
});
