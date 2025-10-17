import { localStorage } from '@kit.ArkTS';

/**
 * Token管理器，用于存储和获取认证Token
 */
export class TokenManager {
  private static readonly TOKEN_KEY: string = 'accessToken';

  /**
   * 设置Token
   * @param token 认证Token
   */
  public static setToken(token: string): void {
    try {
      localStorage.set(TokenManager.TOKEN_KEY, token);
    } catch (error) {
      console.error('设置Token失败:', error);
    }
  }

  /**
   * 获取Token
   * @returns 认证Token
   */
  public static getToken(): string | null {
    try {
      const token = localStorage.get(TokenManager.TOKEN_KEY);
      return token ? token : null;
    } catch (error) {
      console.error('获取Token失败:', error);
      return null;
    }
  }

  /**
   * 清除Token
   */
  public static clearToken(): void {
    try {
      localStorage.remove(TokenManager.TOKEN_KEY);
    } catch (error) {
      console.error('清除Token失败:', error);
    }
  }
}