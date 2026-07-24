import { HttpManager } from '../utils/HttpManager';
import { ApiKeyInfo, ApiKeyCreateForm } from '../model/ApiKeyInfo';

export class ApiKeyAPI {
  private static readonly BASE_URL: string = '/api/v1';

  public static async create(form: ApiKeyCreateForm): Promise<ApiKeyInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<ApiKeyInfo>(`${ApiKeyAPI.BASE_URL}/auth/api-keys`, form as unknown as Record<string, Object>);
  }

  public static async list(): Promise<Array<ApiKeyInfo>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Array<ApiKeyInfo>>(`${ApiKeyAPI.BASE_URL}/auth/api-keys`, undefined);
  }

  public static async delete(id: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${ApiKeyAPI.BASE_URL}/auth/api-keys/${id}`, undefined);
  }
}
