import { pageQuery, uniqueCode, uniqueName } from "./common";
import type { AiModelForm, AiModelQuery } from "../../src/api/ai-conversation/model";

/**
 * 模型创建/更新表单工厂。
 *
 * 对齐后端 ai_conversation.py 的 AiModelCreate 契约：
 * provider_id 为数字（关联供应商主键），能力字段平铺（snake_case 入参，SDK 层为 camelCase）。
 */
export const createAiModelForm = (overrides?: Partial<AiModelForm>): AiModelForm => ({
  providerId: 1,
  modelId: uniqueCode("test_model"),
  displayName: uniqueName("测试模型"),
  inputRate: 1,
  outputRate: 3,
  cachedRate: 0.5,
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

/** 模型分页查询参数工厂 */
export const createAiModelQuery = (overrides?: Partial<AiModelQuery>): AiModelQuery =>
  pageQuery<AiModelQuery>({ ...overrides });
