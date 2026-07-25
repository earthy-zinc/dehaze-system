import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

export class TokenManager {
  private static readonly STORE_NAME: string = 'dehaze_auth';
  private static readonly KEY_SESSION_ID: string = 'sessionId';
  private static readonly KEY_USER_INFO: string = 'userInfo';

  private static prefsInstance: preferences.Preferences = undefined;

  public static async initialize(context: common.UIAbilityContext): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      TokenManager.prefsInstance = await preferences.getPreferences(context, TokenManager.STORE_NAME);
    }
  }

  public static async setSessionId(id: string): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      console.error('[TokenManager] 未初始化，请先调用 initialize');
      return;
    }
    await TokenManager.prefsInstance.put(TokenManager.KEY_SESSION_ID, id);
    await TokenManager.prefsInstance.flush();
  }

  public static getSessionId(): string | null {
    if (TokenManager.prefsInstance === undefined) {
      return null;
    }
    return TokenManager.prefsInstance.getSync(TokenManager.KEY_SESSION_ID, '') as string || null;
  }

  public static async setUserInfo(userInfo: string): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      return;
    }
    await TokenManager.prefsInstance.put(TokenManager.KEY_USER_INFO, userInfo);
    await TokenManager.prefsInstance.flush();
  }

  public static getUserInfo(): string | null {
    if (TokenManager.prefsInstance === undefined) {
      return null;
    }
    return TokenManager.prefsInstance.getSync(TokenManager.KEY_USER_INFO, '') as string || null;
  }

  public static async clearToken(): Promise<void> {
    if (TokenManager.prefsInstance === undefined) {
      return;
    }
    await TokenManager.prefsInstance.delete(TokenManager.KEY_SESSION_ID);
    await TokenManager.prefsInstance.delete(TokenManager.KEY_USER_INFO);
    await TokenManager.prefsInstance.flush();
  }

  public static isLoggedIn(): boolean {
    return !!TokenManager.getSessionId();
  }
}
