import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

/**
 * Token 管理器
 *
 * 基于 @kit.ArkData 的 preferences 实现持久化存储，
 * 存储当前登录用户的 AccessToken / RefreshToken / 用户信息。
 */
export class TokenManager {
  /** preferences 实例名 */
  private static readonly STORE_NAME: string = 'dehaze_auth';
  /** AccessToken 存储 key */
  private static readonly KEY_ACCESS_TOKEN: string = 'accessToken';
  /** RefreshToken 存储 key */
  private static readonly KEY_REFRESH_TOKEN: string = 'refreshToken';
  /** 用户信息存储 key */
  private static readonly KEY_USER_INFO: string = 'userInfo';

  private static prefsInstance: preferences.Preferences = undefined;

  /** 初始化（需在 Ability onCreate 中调用，传入 context） */
  public static async initialize(context: common.UIAbilityContext): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      TokenManager.prefsInstance = await preferences.getPreferences(context, TokenManager.STORE_NAME);
    }
  }

  /** 设置 AccessToken */
  public static async setToken(token: string): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      console.error('[TokenManager] 未初始化，请先调用 initialize');
      return;
    }
    await TokenManager.prefsInstance.put(TokenManager.KEY_ACCESS_TOKEN, token);
    await TokenManager.prefsInstance.flush();
  }

  /** 获取 AccessToken（同步缓存读取） */
  public static getToken(): string | null {
    if (TokenManager.prefsInstance === undefined) {
      return null;
    }
    return TokenManager.prefsInstance.getSync(TokenManager.KEY_ACCESS_TOKEN, '') as string || null;
  }

  /** 设置 RefreshToken */
  public static async setRefreshToken(token: string): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      return;
    }
    await TokenManager.prefsInstance.put(TokenManager.KEY_REFRESH_TOKEN, token);
    await TokenManager.prefsInstance.flush();
  }

  /** 获取 RefreshToken */
  public static getRefreshToken(): string | null {
    if (TokenManager.prefsInstance === undefined) {
      return null;
    }
    return TokenManager.prefsInstance.getSync(TokenManager.KEY_REFRESH_TOKEN, '') as string || null;
  }

  /** 设置用户信息（JSON 字符串） */
  public static async setUserInfo(userInfo: string): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      return;
    }
    await TokenManager.prefsInstance.put(TokenManager.KEY_USER_INFO, userInfo);
    await TokenManager.prefsInstance.flush();
  }

  /** 获取用户信息（JSON 字符串） */
  public static getUserInfo(): string | null {
    if (TokenManager.prefsInstance === undefined) {
      return null;
    }
    return TokenManager.prefsInstance.getSync(TokenManager.KEY_USER_INFO, '') as string || null;
  }

  /** 清除所有认证信息 */
  public static async clearToken(): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      return;
    }
    await TokenManager.prefsInstance.delete(TokenManager.KEY_ACCESS_TOKEN);
    await TokenManager.prefsInstance.delete(TokenManager.KEY_REFRESH_TOKEN);
    await TokenManager.prefsInstance.delete(TokenManager.KEY_USER_INFO);
    await TokenManager.prefsInstance.flush();
  }

  /** 是否已登录（同步） */
  public static isLoggedIn(): boolean {
    return !!TokenManager.getToken();
  }
}
