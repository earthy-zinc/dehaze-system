import { router } from '@kit.ArkUI';
import { promptAction } from '@kit.ArkUI';
import { AuthAPI, LoginData, CaptchaResult, AuthUserInfo } from 'dehaze-sdk-harmony';
import { AuthState } from '../common/AppState';

/**
 * 登录页面
 *
 * 功能（对应认证管理模块前端实现设计）：
 * - 用户名 + 密码 + 图形验证码登录
 * - 验证码图片点击刷新
 * - 登录成功后获取用户信息并跳转首页
 */
@Entry
@Component
struct Login {
  @State username: string = '';
  @State password: string = '';
  @State captchaCode: string = '';
  @State captchaKey: string = '';
  @State captchaBase64: string = '';
  @State captchaLoading: boolean = false;
  @State loading: boolean = false;
  @State passwordVisible: boolean = false;

  aboutToAppear(): void {
    this.refreshCaptcha();
  }

  /** 刷新验证码 */
  async refreshCaptcha(): Promise<void> {
    this.captchaLoading = true;
    this.captchaBase64 = '';
    try {
      const result: CaptchaResult = await AuthAPI.getCaptcha();
      this.captchaKey = result.captchaKey;
      // 后端返回 Base64（不含 data: 前缀），Image 组件需要完整 data URL
      this.captchaBase64 = `data:image/png;base64,${result.captchaBase64}`;
    } catch (e) {
      promptAction.showToast({ message: '获取验证码失败，请检查网络', duration: 2000 });
    } finally {
      this.captchaLoading = false;
    }
  }

  /** 表单校验 */
  validateForm(): boolean {
    if (!this.username.trim()) {
      promptAction.showToast({ message: '请输入用户名' });
      return false;
    }
    if (!this.password.trim()) {
      promptAction.showToast({ message: '请输入密码' });
      return false;
    }
    if (!this.captchaCode.trim()) {
      promptAction.showToast({ message: '请输入验证码' });
      return false;
    }
    return true;
  }

  /** 提交登录 */
  async handleLogin(): Promise<void> {
    if (!this.validateForm()) {
      return;
    }
    this.loading = true;

    const loginData: LoginData = {
      username: this.username.trim(),
      password: this.password.trim(),
      captchaKey: this.captchaKey,
      captchaCode: this.captchaCode.trim(),
    };

    try {
      const result = await AuthAPI.login(loginData);

      // 登录成功，获取用户完整信息
      let userInfo: AuthUserInfo | undefined = undefined;
      try {
        userInfo = await AuthAPI.getMe();
      } catch (e) {
        // 获取用户信息失败不影响登录
      }

      // 更新全局状态
      await AuthState.onLoginSuccess(result, userInfo);

      promptAction.showToast({ message: '登录成功' });

      setTimeout(() => {
        router.replaceUrl({ url: 'pages/Home' });
      }, 800);
    } catch (e) {
      const err = e as { code?: string; message?: string };
      const msg = err?.message || '登录失败，请重试';

      // 验证码错误时刷新验证码
      if (err?.code === 'A0213' || err?.code === 'A0214' || msg.includes('验证码')) {
        promptAction.showToast({ message: msg });
        this.captchaCode = '';
        this.refreshCaptcha();
      } else {
        promptAction.showToast({ message: msg, duration: 2500 });
      }
    } finally {
      this.loading = false;
    }
  }

  build() {
    Column() {
      // 顶部装饰区
      Column() {
        Text('图像去雾系统')
          .fontSize(28)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333333')
          .margin({ bottom: 8 })

        Text('Dehaze System')
          .fontSize(14)
          .fontColor('#999999')
      }
      .width('100%')
      .height(180)
      .justifyContent(FlexAlign.Center)

      // 登录卡片
      Column() {
        // 用户名
        TextInput({ text: this.username, placeholder: '请输入用户名' })
          .height(48)
          .fontSize(16)
          .borderRadius(8)
          .backgroundColor('#F5F5F5')
          .onChange((value: string) => {
            this.username = value;
          })
          .margin({ bottom: 16 })

        // 密码
        TextInput({ text: this.password, placeholder: '请输入密码' })
          .height(48)
          .fontSize(16)
          .borderRadius(8)
          .backgroundColor('#F5F5F5')
          .type(this.passwordVisible ? InputType.Normal : InputType.Password)
          .onChange((value: string) => {
            this.password = value;
          })
          .margin({ bottom: 16 })

        // 验证码
        Row() {
          TextInput({ text: this.captchaCode, placeholder: '验证码' })
            .height(48)
            .fontSize(16)
            .borderRadius(8)
            .backgroundColor('#F5F5F5')
            .layoutWeight(1)
            .onChange((value: string) => {
              this.captchaCode = value;
            })

          // 验证码图片
          Image(this.captchaBase64)
            .height(48)
            .width(120)
            .borderRadius(8)
            .backgroundColor('#F0F0F0')
            .objectFit(ImageFit.Contain)
            .margin({ left: 12 })
            .onClick(() => {
              this.refreshCaptcha();
            })
        }
        .width('100%')
        .margin({ bottom: 24 })

        // 登录按钮
        Button(this.loading ? '登录中...' : '登 录')
          .width('100%')
          .height(48)
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .backgroundColor(this.loading ? '#CCCCCC' : '#389e3c')
          .fontColor('#FFFFFF')
          .borderRadius(8)
          .enabled(!this.loading)
          .onClick(() => {
            this.handleLogin();
          })
      }
      .width('85%')
      .padding(24)
      .backgroundColor('#FFFFFF')
      .borderRadius(16)
      .shadow({ radius: 20, color: 'rgba(0,0,0,0.08)', offsetX: 0, offsetY: 4 })

      // 底部提示
      Column() {
        Text('图像去雾系统 v1.0.0')
          .fontSize(13)
          .fontColor('#999999')
          .margin({ top: 24 })
      }
      .width('100%')
      .layoutWeight(1)
      .justifyContent(FlexAlign.End)
      .padding({ bottom: 40 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F7FA')
    .alignItems(HorizontalAlign.Center)
  }
}
