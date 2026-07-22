import { TokenManager } from 'dehaze-sdk-harmony';
import { AuthUserInfo, LoginResult } from 'dehaze-sdk-harmony';

/**
 * 全局状态管理
 *
 * 基于 AppStorage 实现跨页面状态共享，管理：
 * - 登录态（isLoggedIn）
 * - 当前用户信息（userInfo）
 * - 去雾处理流程中的上下文（原图 URL、选中算法、处理结果、GT 参考图、算法参数）
 */

/** AppStorage 键名 */
export class AppKeys {
  /** 是否已登录 */
  static readonly IS_LOGGED_IN: string = 'isLoggedIn';
  /** 当前用户信息（JSON 字符串） */
  static readonly USER_INFO: string = 'userInfo';
  /** 用户昵称 */
  static readonly NICKNAME: string = 'nickname';
  /** 用户头像 URL */
  static readonly AVATAR: string = 'avatar';
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
  /** 去雾处理-GT 参考图 URL（必须是 clean 图） */
  static readonly GT_URL: string = 'gtUrl';
  /** 去雾处理-算法参数（JSON 字符串） */
  static readonly ALGORITHM_PARAMS: string = 'algorithmParams';
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
        AppStorage.setOrCreate<string>(AppKeys.AVATAR, userInfo.avatar || '');
      } catch (e) {
        AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
        AppStorage.setOrCreate<string>(AppKeys.AVATAR, '');
      }
    } else {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
      AppStorage.setOrCreate<string>(AppKeys.AVATAR, '');
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
      AppStorage.setOrCreate<string>(AppKeys.AVATAR, userInfo.avatar || '');
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
    AppStorage.setOrCreate<string>(AppKeys.AVATAR, '');
    // 清除处理上下文
    ProcessingContext.clear();
  }
}

/**
 * 去雾处理上下文管理
 */
export class ProcessingContext {
  /** 设置原图，可同时设置 GT 参考图（clean 图） */
  static setOriginal(url: string, uri?: string, gtUrl?: string): void {
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URL, url);
    if (uri) {
      AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URI, uri);
    }
    if (gtUrl !== undefined) {
      AppStorage.setOrCreate<string>(AppKeys.GT_URL, gtUrl);
    }
  }

  /** 设置 GT 参考图（必须是 clean 图） */
  static setGtUrl(gtUrl: string): void {
    AppStorage.setOrCreate<string>(AppKeys.GT_URL, gtUrl);
  }

  /** 获取原图 URL */
  static getOriginalUrl(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URL) || '';
  }

  /** 获取原图文件 URI */
  static getOriginalUri(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URI) || '';
  }

  /** 获取 GT 参考图 URL */
  static getGtUrl(): string {
    return AppStorage.get<string>(AppKeys.GT_URL) || '';
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

  /** 设置算法参数（JSON 字符串） */
  static setAlgorithmParams(params: string): void {
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_PARAMS, params);
  }

  /** 获取算法参数（JSON 字符串） */
  static getAlgorithmParams(): string {
    return AppStorage.get<string>(AppKeys.ALGORITHM_PARAMS) || '';
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

  /** 清空处理上下文 */
  static clear(): void {
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URL, '');
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URI, '');
    AppStorage.setOrCreate<number>(AppKeys.ALGORITHM_ID, 0);
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_NAME, '');
    AppStorage.setOrCreate<string>(AppKeys.RESULT_URL, '');
    AppStorage.setOrCreate<number>(AppKeys.PROCESSING_TIME, 0);
    AppStorage.setOrCreate<string>(AppKeys.GT_URL, '');
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_PARAMS, '');
  }
}
