import { TokenManager } from 'dehaze-sdk-harmony';
import { AuthUserInfo, LoginResult } from 'dehaze-sdk-harmony';

export class AppKeys {
  static readonly IS_LOGGED_IN: string = 'isLoggedIn';
  static readonly USER_INFO: string = 'userInfo';
  static readonly NICKNAME: string = 'nickname';
  static readonly AVATAR: string = 'avatar';
  static readonly ORIGINAL_URL: string = 'originalUrl';
  static readonly ORIGINAL_URI: string = 'originalUri';
  static readonly ALGORITHM_ID: string = 'algorithmId';
  static readonly ALGORITHM_NAME: string = 'algorithmName';
  static readonly RESULT_URL: string = 'resultUrl';
  static readonly PROCESSING_TIME: string = 'processingTime';
  static readonly GT_URL: string = 'gtUrl';
  static readonly ALGORITHM_PARAMS: string = 'algorithmParams';
}

export class AuthState {
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

  static async onLoginSuccess(result: LoginResult, userInfo?: AuthUserInfo): Promise<void> {
    await TokenManager.setSessionId(result.sessionId);

    AppStorage.setOrCreate<boolean>(AppKeys.IS_LOGGED_IN, true);

    if (userInfo) {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, userInfo.nickname || userInfo.username);
      AppStorage.setOrCreate<string>(AppKeys.AVATAR, userInfo.avatar || '');
      await TokenManager.setUserInfo(JSON.stringify(userInfo));
    } else if (result.user) {
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, result.user.nickname || result.user.username);
    }
  }

  static async onLogout(): Promise<void> {
    await TokenManager.clearToken();
    AppStorage.setOrCreate<boolean>(AppKeys.IS_LOGGED_IN, false);
    AppStorage.setOrCreate<string>(AppKeys.NICKNAME, '');
    AppStorage.setOrCreate<string>(AppKeys.AVATAR, '');
    ProcessingContext.clear();
  }
}

export class ProcessingContext {
  static setOriginal(url: string, uri?: string, gtUrl?: string): void {
    AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URL, url);
    if (uri) {
      AppStorage.setOrCreate<string>(AppKeys.ORIGINAL_URI, uri);
    }
    if (gtUrl !== undefined) {
      AppStorage.setOrCreate<string>(AppKeys.GT_URL, gtUrl);
    }
  }

  static setGtUrl(gtUrl: string): void {
    AppStorage.setOrCreate<string>(AppKeys.GT_URL, gtUrl);
  }

  static getOriginalUrl(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URL) || '';
  }

  static getOriginalUri(): string {
    return AppStorage.get<string>(AppKeys.ORIGINAL_URI) || '';
  }

  static getGtUrl(): string {
    return AppStorage.get<string>(AppKeys.GT_URL) || '';
  }

  static setAlgorithm(id: number, name: string): void {
    AppStorage.setOrCreate<number>(AppKeys.ALGORITHM_ID, id);
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_NAME, name);
  }

  static getAlgorithmId(): number {
    return AppStorage.get<number>(AppKeys.ALGORITHM_ID) || 0;
  }

  static getAlgorithmName(): string {
    return AppStorage.get<string>(AppKeys.ALGORITHM_NAME) || '';
  }

  static setAlgorithmParams(params: string): void {
    AppStorage.setOrCreate<string>(AppKeys.ALGORITHM_PARAMS, params);
  }

  static getAlgorithmParams(): string {
    return AppStorage.get<string>(AppKeys.ALGORITHM_PARAMS) || '';
  }

  static setResult(url: string, time?: number): void {
    AppStorage.setOrCreate<string>(AppKeys.RESULT_URL, url);
    if (time !== undefined) {
      AppStorage.setOrCreate<number>(AppKeys.PROCESSING_TIME, time);
    }
  }

  static getResultUrl(): string {
    return AppStorage.get<string>(AppKeys.RESULT_URL) || '';
  }

  static getProcessingTime(): number {
    return AppStorage.get<number>(AppKeys.PROCESSING_TIME) || 0;
  }

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
