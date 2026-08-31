import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  AiModelForm,
  AiModelQuery,
  AiModelType,
  AiModelUpdateForm,
  AiModelVO,
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

  /** 启用模型列表（按登录用户 VIP 等级过滤，无特殊权限） */
  static listEnabledModels(modelType?: AiModelType) {
    return request<AiModelVO[]>({
      url: "/api/v1/ai/models/enabled",
      method: "get",
      params: modelType ? { modelType } : undefined,
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

  /** 更新模型（管理员，路径参数为 model_id 业务标识） */
  static updateModel(modelId: string, data: AiModelUpdateForm) {
    return request<AiModelVO>({
      url: `/api/v1/ai/models/${modelId}`,
      method: "put",
      data,
    });
  }

  /** 删除模型（管理员，逻辑删除，model_id 不可复用） */
  static deleteModel(modelId: string) {
    return request({
      url: `/api/v1/ai/models/${modelId}`,
      method: "delete",
    });
  }

  // ==================== 模型用户售价（价格版本） ====================

  /** 价格版本分页列表（管理员） */
  static listPrices(modelId: string, query?: ModelPriceQuery) {
    return request<PageResult<ModelPriceVO[]>>({
      url: `/api/v1/ai/models/${modelId}/prices`,
      method: "get",
      params: query,
    });
  }

  /** 新增价格版本（管理员，同模型同供应商版本号递增） */
  static createPrice(modelId: string, data: ModelPriceForm) {
    return request<ModelPriceVO>({
      url: `/api/v1/ai/models/${modelId}/prices`,
      method: "post",
      data,
    });
  }

  /** 更新价格版本（管理员，仅单价单位/生效时间/状态） */
  static updatePrice(modelId: string, priceId: number, data: ModelPriceUpdateForm) {
    return request<ModelPriceVO>({
      url: `/api/v1/ai/models/${modelId}/prices/${priceId}`,
      method: "put",
      data,
    });
  }

  /** 删除价格版本（管理员，主表与档位明细一并逻辑删除） */
  static deletePrice(modelId: string, priceId: number) {
    return request({
      url: `/api/v1/ai/models/${modelId}/prices/${priceId}`,
      method: "delete",
    });
  }
}

export default AiModelAPI;
