import { ApiKeyCreateForm } from "@/api/api-key/model";
import { uniqueName } from "./common";

/**
 * 创建 API Key 的表单（基础字段）。
 * 治理参数（dailyQuota/monthlyQuota/rpmLimit/modelWhitelist）由后端支持，
 * 但 SDK 的 ApiKeyCreateForm 类型未声明，故在 createGovernedApiKeyForm 中以透传对象提供。
 */
export function createApiKeyForm(overrides: Partial<ApiKeyCreateForm> = {}): ApiKeyCreateForm {
  return {
    name: uniqueName("测试密钥"),
    ...overrides,
  };
}

/**
 * 创建携带治理参数的 API Key 表单。
 * 治理参数后端已支持（sys_api_key.daily_quota/monthly_quota/rpm_limit/model_whitelist），
 * SDK 透传 data 提交，测试侧用宽松对象类型覆盖（避免修改 src 类型）。
 */
export function createGovernedApiKeyForm(
  overrides: {
    name?: string;
    expiresAt?: string;
    dailyQuota?: number;
    monthlyQuota?: number;
    rpmLimit?: number;
    modelWhitelist?: string[];
  } = {}
): Record<string, unknown> {
  return {
    name: uniqueName("治理密钥"),
    ...overrides,
  };
}
