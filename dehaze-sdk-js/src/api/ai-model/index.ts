import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  AiModelForm,
  AiModelQuery,
  AiModelType,
  AiModelUpdateForm,
  AiModelVO,
  AiUserPreference,
  AiUserPreferenceForm,
  ModelPriceForm,
  ModelPriceQuery,
  ModelPriceUpdateForm,
  ModelPriceVO,
} from "./model";

/**
 * AI 模型管理 API
 *
 * 内部 API（`/api/v1/ai/models`），除 `enabled` 列表外均需 `ai:model:manage` 权限
 * （由后端拦截，越权返回 403 A0301）。
 *
 * 模型与供应商是多对一关系，模型以 `model_id` 字符串为业务主键（非自增 id），
 * 删除为逻辑删除且 `model_id` 不可复用。价格按版本管理，新增即生成新版本。
 */
class AiModelAPI {
  // ==================== 模型 ====================

  /** 模型分页列表（管理员，权限 ai:model:manage） */
  static listModels(query?: AiModelQuery) {
    return request<PageResult<AiModelVO[]>>({
      url: "/api/v1/ai/models",
      method: "get",
      params: query,
    });
  }

  /** 启用模型列表（按登录用户 VIP 等级过滤，无特殊权限；taskTags 逗号分隔、含任一即命中） */
  static listEnabledModels(modelType?: AiModelType, taskTags?: string | null) {
    const params: Record<string, string> = {};
    if (modelType) params.modelType = modelType;
    if (taskTags) params.taskTags = taskTags;
    return request<AiModelVO[]>({
      url: "/api/v1/ai/models/enabled",
      method: "get",
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /** 新增模型（管理员，embedding 类型必须传 dimension） */
  static createModel(data: AiModelForm) {
    return request<AiModelVO>({
      url: "/api/v1/ai/models",
      method: "post",
      data,
    });
  }

  /** 更新模型（管理员，model_id 经 query 参数传递） */
  static updateModel(modelId: string, data: AiModelUpdateForm) {
    return request<AiModelVO>({
      url: "/api/v1/ai/models",
      method: "put",
      params: { modelId },
      data,
    });
  }

  /** 删除模型（管理员，逻辑删除，model_id 不可复用） */
  static deleteModel(modelId: string) {
    return request({
      url: "/api/v1/ai/models",
      method: "delete",
      params: { modelId },
    });
  }

  /** 模型可用性测试（管理员）：发最小真实推理请求并落库结果 */
  static testModel(modelId: string) {
    return request<{
      success: boolean;
      latencyMs: number | null;
      error: string | null;
    }>({
      url: "/api/v1/ai/models/test",
      method: "post",
      params: { modelId },
    });
  }

  // ==================== 模型用户售价（价格版本） ====================

  /** 价格版本分页列表（管理员，model_id 经 query 参数传递） */
  static listPrices(modelId: string, query?: ModelPriceQuery) {
    return request<PageResult<ModelPriceVO[]>>({
      url: "/api/v1/ai/models/prices",
      method: "get",
      params: { modelId, ...query },
    });
  }

  /** 新增价格版本（管理员，同模型同供应商版本号递增；modelId 随请求体提交） */
  static createPrice(data: ModelPriceForm) {
    return request<ModelPriceVO>({
      url: "/api/v1/ai/models/prices",
      method: "post",
      data,
    });
  }

  /** 更新价格版本（管理员，仅单价单位/生效时间/状态） */
  static updatePrice(priceId: number, data: ModelPriceUpdateForm) {
    return request<ModelPriceVO>({
      url: `/api/v1/ai/models/prices/${priceId}`,
      method: "put",
      data,
    });
  }

  /** 删除价格版本（管理员，主表与档位明细一并逻辑删除） */
  static deletePrice(priceId: number) {
    return request({
      url: `/api/v1/ai/models/prices/${priceId}`,
      method: "delete",
    });
  }
}

/**
 * 用户偏好默认模型 API
 *
 * 用户级偏好，仅决定新建会话的默认模型；已有会话继续使用自身模型。
 */
class AiUserPreferenceAPI {
  /** 查询当前用户偏好（无记录返回 defaultModel: null） */
  static get() {
    return request<AiUserPreference>({
      url: "/api/v1/ai/user-preference",
      method: "get",
    });
  }

  /** 保存偏好（defaultModel 传 null 表示清除，回落平台默认模型） */
  static update(data: AiUserPreferenceForm) {
    return request({
      url: "/api/v1/ai/user-preference",
      method: "put",
      data,
    });
  }
}

export default AiModelAPI;
export { AiUserPreferenceAPI };
