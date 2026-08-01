import request from "@/utils/request";
import { ApiKeyCreateForm, ApiKeyVO } from "./model";

class ApiKeyAPI {
  static create(data: ApiKeyCreateForm) {
    return request<ApiKeyVO>({
      url: "/api/v1/auth/api-keys",
      method: "post",
      data: data,
    });
  }

  static list() {
    return request<ApiKeyVO[]>({
      url: "/api/v1/auth/api-keys",
      method: "get",
    });
  }

  /**
   * 吊销 API 密钥（DELETE /{id}）
   * 内部语义：设 revoked_at=now()，不再物理删除。
   */
  static delete(id: number) {
    return request({
      url: "/api/v1/auth/api-keys/" + id,
      method: "delete",
    });
  }
}

export default ApiKeyAPI;
