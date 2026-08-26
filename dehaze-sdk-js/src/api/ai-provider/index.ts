import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  ConnectionTestResult,
  ProviderCreateForm,
  ProviderKeyCreateForm,
  ProviderKeyUpdateForm,
  ProviderKeyVO,
  ProviderPageQuery,
  ProviderUpdateForm,
  ProviderVO,
  UsageStatQuery,
  UsageStatsVO,
} from "./model";

/**
 * AI 模型供应商与 API Key 管理 API
 *
 * 内部 API（`/api/v1/ai/providers`），供应商与 Key 相关为管理员接口，
 * 需 `ai:model:manage` 权限（由后端拦截，403 由后端返回）；`enabled` 列表无需特殊权限。
 *
 * 注意：Key 创建/查询响应均不含明文，仅返回 keyPrefix 等展示字段。
 */
class AiProviderAPI {
  // ==================== 供应商 ====================

  /** 供应商分页列表（管理员，权限 ai:model:manage） */
  static listProviders(query?: ProviderPageQuery) {
    return request<PageResult<ProviderVO[]>>({
      url: "/api/v1/ai/providers",
      method: "get",
      params: query,
    });
  }

  /** 启用供应商列表（供调用方下拉选择，无特殊权限） */
  static listEnabledProviders() {
    return request<ProviderVO[]>({
      url: "/api/v1/ai/providers/enabled",
      method: "get",
    });
  }

  /** 新增供应商（管理员，保存后后端异步触发连通性测试，结果仅提示不阻断） */
  static createProvider(data: ProviderCreateForm) {
    return request<ProviderVO>({
      url: "/api/v1/ai/providers",
      method: "post",
      data,
    });
  }

  /** 更新供应商（管理员） */
  static updateProvider(id: number, data: ProviderUpdateForm) {
    return request<ProviderVO>({
      url: `/api/v1/ai/providers/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除供应商（管理员，软删除） */
  static deleteProvider(id: number) {
    return request({
      url: `/api/v1/ai/providers/${id}`,
      method: "delete",
    });
  }

  // ==================== API Key ====================

  /** 供应商 API Key 列表（管理员） */
  static listKeys(providerId: number) {
    return request<ProviderKeyVO[]>({
      url: `/api/v1/ai/providers/${providerId}/keys`,
      method: "get",
    });
  }

  /** 新增 API Key（管理员，响应不含明文） */
  static createKey(providerId: number, data: ProviderKeyCreateForm) {
    return request<ProviderKeyVO>({
      url: `/api/v1/ai/providers/${providerId}/keys`,
      method: "post",
      data,
    });
  }

  /** 更新 API Key（管理员） */
  static updateKey(providerId: number, keyId: number, data: ProviderKeyUpdateForm) {
    return request<ProviderKeyVO>({
      url: `/api/v1/ai/providers/${providerId}/keys/${keyId}`,
      method: "put",
      data,
    });
  }

  /** 删除 API Key（管理员，软删除） */
  static deleteKey(providerId: number, keyId: number) {
    return request({
      url: `/api/v1/ai/providers/${providerId}/keys/${keyId}`,
      method: "delete",
    });
  }

  // ==================== 连通性测试 / 熔断 ====================

  /** 供应商连通性测试（管理员，同步返回测试结果） */
  static testConnection(providerId: number) {
    return request<ConnectionTestResult>({
      url: `/api/v1/ai/providers/${providerId}/test-connection`,
      method: "post",
    });
  }

  /** 手动解除供应商熔断（管理员） */
  static closeCircuit(providerId: number) {
    return request({
      url: `/api/v1/ai/providers/${providerId}/circuit/close`,
      method: "post",
    });
  }

  // ==================== 运营统计 ====================

  /** 运营统计（供应商健康看板/模型用量分布/降级与故障统计，管理员，需 ai:model:manage） */
  static getUsageStats(query?: UsageStatQuery) {
    return request<UsageStatsVO>({
      url: "/api/v1/ai/usage/stats",
      method: "get",
      params: query,
    });
  }
}

export default AiProviderAPI;
