import { HttpManager } from '../utils/HttpManager';
import { LoginData, LoginResult, CaptchaResult, AuthUserInfo } from '../model/AuthInfo';
import { RegisterData } from '../model/AuthInfo';

export class AuthAPI {
  private static readonly BASE_URL: string = '/api/v1';

  public static async getCaptcha(): Promise<CaptchaResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<CaptchaResult>(`${AuthAPI.BASE_URL}/auth/captcha`, undefined);
  }

  public static async login(data: LoginData): Promise<LoginResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<LoginResult>(`${AuthAPI.BASE_URL}/auth/login`, data as unknown as Record<string, Object>);
  }

  public static async register(data: RegisterData): Promise<LoginResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<LoginResult>(`${AuthAPI.BASE_URL}/auth/register`, data as unknown as Record<string, Object>);
  }

  public static async logout(): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.post<void>(`${AuthAPI.BASE_URL}/auth/logout`, undefined);
  }

  public static async getMe(): Promise<AuthUserInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<AuthUserInfo>(`${AuthAPI.BASE_URL}/auth/me`, undefined);
  }
}
