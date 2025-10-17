import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { LoginData, LoginResult, CaptchaResult } from '../model/AuthInfo';

/**
 * 认证相关API服务接口
 */
export class AuthAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 登录API
   *
   * @param data 登录参数
   * @returns 登录结果
   */
  public static async login(data: LoginData): Promise<Result<LoginResult>> {
    const formData = new FormData();
    formData.append('username', data.username);
    formData.append('password', data.password);
    formData.append('captchaKey', data.captchaKey || '');
    formData.append('captchaCode', data.captchaCode || '');
    
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<LoginResult>>(`${AuthAPI.BASE_URL}/auth/login`, formData);
  }

  /**
   * 注销API
   *
   * @returns 注销结果
   */
  public static async logout(): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${AuthAPI.BASE_URL}/auth/logout`);
  }

  /**
   * 获取验证码
   *
   * @returns 验证码结果
   */
  public static async getCaptcha(): Promise<Result<CaptchaResult>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<CaptchaResult>>(`${AuthAPI.BASE_URL}/auth/captcha`);
  }
}