import { HttpManager } from '../utils/HttpManager';
import { LoginData, LoginResult, CaptchaResult, AuthUserInfo } from '../model/AuthInfo';

/**
 * 认证相关 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/auth.go）：
 * - GET  /api/v1/auth/captcha   获取验证码
 * - POST /api/v1/auth/login      登录
 * - POST /api/v1/auth/logout     登出
 * - GET  /api/v1/auth/me         获取当前用户信息
 * - POST /api/v1/auth/refresh    刷新 Token
 */
export class AuthAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 获取验证码 */
  public static async getCaptcha(): Promise<CaptchaResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<CaptchaResult>(`${AuthAPI.BASE_URL}/auth/captcha`, undefined);
  }

  /** 登录 */
  public static async login(data: LoginData): Promise<LoginResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<LoginResult>(`${AuthAPI.BASE_URL}/auth/login`, data as unknown as Record<string, Object>);
  }

  /** 登出 */
  public static async logout(): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.post<void>(`${AuthAPI.BASE_URL}/auth/logout`, undefined);
  }

  /** 获取当前用户信息（/auth/me） */
  public static async getMe(): Promise<AuthUserInfo> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<AuthUserInfo>(`${AuthAPI.BASE_URL}/auth/me`, undefined);
  }

  /** 刷新 Token */
  public static async refreshToken(refreshToken: string): Promise<LoginResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<LoginResult>(`${AuthAPI.BASE_URL}/auth/refresh`, { refreshToken } as Record<string, Object>);
  }
}
