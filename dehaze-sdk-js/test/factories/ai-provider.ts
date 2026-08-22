import { pageQuery, uniqueCode, uniqueName } from "./common";
import type {
  ProviderCreateForm,
  ProviderKeyCreateForm,
  ProviderKeyUpdateForm,
  ProviderPageQuery,
  ProviderUpdateForm,
} from "../../src/api/ai-provider/model";

/** 供应商创建表单工厂（provider_code 前缀 test_prov_） */
export const createProviderForm = (
  overrides?: Partial<ProviderCreateForm>
): ProviderCreateForm => ({
  providerCode: uniqueCode("test_prov"),
  displayName: uniqueName("测试供应商"),
  apiBaseUrl: "https://api.openai.com/v1",
  protocolType: "openai_compat",
  authType: "bearer",
  sortOrder: 0,
  healthCheckEnabled: 0,
  status: 1,
  ...overrides,
});

/** 供应商更新表单工厂 */
export const createProviderUpdateForm = (
  overrides?: Partial<ProviderUpdateForm>
): ProviderUpdateForm => ({
  displayName: uniqueName("更新供应商"),
  ...overrides,
});

/** 供应商分页查询参数工厂 */
export const createProviderQuery = (overrides?: Partial<ProviderPageQuery>): ProviderPageQuery =>
  pageQuery<ProviderPageQuery>({ ...overrides });

/** API Key 创建表单工厂（key 明文仅供创建时提交，响应不含明文） */
export const createProviderKeyForm = (
  overrides?: Partial<ProviderKeyCreateForm>
): ProviderKeyCreateForm => ({
  name: uniqueName("测试Key"),
  key: `sk-test-${uniqueCode("k")}`,
  priority: 0,
  weight: 1,
  status: 1,
  ...overrides,
});

/** API Key 更新表单工厂 */
export const createProviderKeyUpdateForm = (
  overrides?: Partial<ProviderKeyUpdateForm>
): ProviderKeyUpdateForm => ({
  name: uniqueName("更新Key"),
  ...overrides,
});
