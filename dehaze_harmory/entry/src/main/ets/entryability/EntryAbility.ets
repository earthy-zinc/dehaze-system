import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window, router } from '@kit.ArkUI';
import { DehazeSDK, Builder, TokenManager } from 'dehaze-sdk-harmony';
import { AuthState } from '../common/AppState';

const DOMAIN = 0x0000;
const TAG = 'EntryAbility';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    try {
      this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    } catch (err) {
      hilog.error(DOMAIN, TAG, 'Failed to set colorMode. Cause: %{public}s', JSON.stringify(err));
    }

    // 初始化 SDK
    const baseUrl = 'http://127.0.0.1:8989';
    DehazeSDK.initialize(this.context, new Builder().setBaseUrl(baseUrl).setDebug(true)).then(() => {
      hilog.info(DOMAIN, TAG, 'DehazeSDK initialized');

      // 设置 Token 失效回调（跳转登录页）
      DehazeSDK.setTokenInvalidCallback(() => {
        hilog.info(DOMAIN, TAG, 'Token invalid, redirecting to login');
        AuthState.onLogout();
        router.replaceUrl({ url: 'pages/Login' });
      });

      // 从持久化存储恢复登录态
      AuthState.initFromStorage();
    }).catch((err: Error) => {
      hilog.error(DOMAIN, TAG, 'Failed to init SDK: %{public}s', err.message);
    });

    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onCreate');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageCreate');

    // 根据登录态决定加载页面
    const isLoggedIn = TokenManager.isLoggedIn();
    const page = isLoggedIn ? 'pages/Home' : 'pages/Login';

    windowStage.loadContent(page, (err) => {
      if (err.code) {
        hilog.error(DOMAIN, TAG, 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, TAG, 'Succeeded in loading the content: %{public}s', page);
    });
  }

  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onBackground');
  }
}
