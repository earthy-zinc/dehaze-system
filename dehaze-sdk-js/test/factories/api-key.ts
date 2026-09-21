import { ApiKeyCreateForm } from "@/api/api-key/model";
import { uniqueName } from "./common";

export function createApiKeyForm(overrides: Partial<ApiKeyCreateForm> = {}): ApiKeyCreateForm {
  return {
    name: uniqueName("测试密钥"),
    ...overrides,
  };
}
