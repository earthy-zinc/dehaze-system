import { TokenManager } from 'dehaze-sdk-harmony';
import { AuthUserInfo, LoginResult } from 'dehaze-sdk-harmony';

/**
 * 全局状态管理
 *
 * 基于 AppStorage 实现跨页面状态共享，管理：
 * - 登录态（isLoggedIn）
 * - 当前用户信息（userInfo）
 * - 去雾处理流程中的上下文（原图 URL、选中算法、处理结果）
 */

/** AppStorage 键名 */
export class AppKeys {
  /** 是否已登录 */
  static readonly IS_LOGGED_IN: string = 'isLoggedIn';
  /** 当前用户信息（JSON 字符串） */
  static readonly USER_INFO: string = 'userInfo';
  /** 用户昵称 */
  static readonly NICKNAME: string = 'nickname';
  /** 去雾处理-原图 URL */
  static readonly ORIGINAL_URL: string = 'originalUrl';
  /** 去雾处理-原图文件 URI（本地） */
  static readonly ORIGINAL_URI: string = 'originalUri';
  /** 去雾处理-选中算法 ID */
  static readonly ALGORITHM_ID: string = 'algorithmId';
  /** 去雾处理-选中算法名称 */
  static readonly ALGORITHM_NAME: string = 'algorithmName';
  /** 去雾处理-结果图 URL */
  static readonly RESULT_URL: string = 'resultUrl';
  /** 去雾处理-处理耗时 */
  static readonly PROCESSING_TIME: string = 'processingTime';
}

/**
 * 认证状态管理
 */
export class AuthState {
  /** 初始化登录态（从持久化存储恢复） */
  static initFromStorage(): void {
    const isLoggedIn = TokenManager.isLoggedIn();
    AppStorage.setOrCreate<boolean>(AppKeys.IS_LOGGED_IN, isLoggedIn);

    const userInfoStr = TokenManager.getUserInfo();
    if (userInfoStr) {
      try {
        const userInfo: AuthUserInfo = JSON.parse(userInfoStr);
        AppStorage.setOrCreate<string>(AppKeys.NICKNAME, userInfo.nickname || userInfo.username);
      } catch (e) {
        AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
      }
    } else {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
    }
  }

  /** 登录成功后设置状态 */
  static async onLoginSuccess(result: LoginResult, userInfo?: AuthUserInfo): Promise<void> {
    // 持久化 Token
    await TokenManager.setToken(result.accessToken);
    await TokenManager.setRefreshToken(result.refreshToken);

    // 更新 AppStorage
    AppStorage.setOrCreate<boolean>(AppKeys.IS_LOGGED_IN, true);

    if (userInfo) {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, userInfo.nickname || userInfo.username);
      await TokenManager.setUserInfo(JSON.stringify(userInfo));
    } else if (result.user) {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, result.user.nickname || result.user.username);
    }
  }

  /** 登出，清除所有状态 */
  static async onLogout(): Promise<void> {
    await TokenManager.clearToken();
    AppStorage.setOrCreate<boolean>(AppKeys.IS_LOGGED_IN, false);
    AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
    // 清除处理上下文
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URL, '');
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URI, '');
    AppStorage.setOrCreate<number>(AppKeys.ALGORITHM_ID, 0);
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_NAME, '');
    AppStorage.setOrCreate<string>(AppKeys.RESULT_URL, '');
  }
}

/**
 * 去雾处理上下文管理
 */
export class ProcessingContext {
  /** 设置原图 */
  static setOriginal(url: string, uri?: string): void {
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URL, url);
    if (uri) {
      AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URI, uri);
    }
  }

  /** 获取原图 URL */
  static getOriginalUrl(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URL) || '';
  }

  /** 获取原图文件 URI */
  static getOriginalUri(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URI) || '';
  }

  /** 设置选中的算法 */
  static setAlgorithm(id: number, name: string): void {
    AppStorage.setOrCreate<number>(AppKeys.ALGORITHM_ID, id);
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_NAME, name);
  }

  /** 获取算法 ID */
  static getAlgorithmId(): number {
    return AppStorage.get<number>(AppKeys.ALGORITHM_ID) || 0;
  }

  /** 获取算法名称 */
  static getAlgorithmName(): string {
    return AppStorage.get<string>(AppKeys.ALGORITHM_NAME) || '';
  }

  /** 设置处理结果 */
  static setResult(url: string, time?: number): void {
    AppStorage.setOrCreate<string>(AppKeys.RESULT_URL, url);
    if (time !== undefined) {
      AppStorage.setOrCreate<number>(AppKeys.PROCESSING_TIME, time);
    }
  }

  /** 获取结果图 URL */
  static getResultUrl(): string {
    return AppStorage.get<string>(AppKeys.RESULT_URL) || '';
  }

  /** 获取处理耗时 */
  static getProcessingTime(): number {
    return AppStorage.get<number>(AppKeys.PROCESSING_TIME) || 0;
  }
}
