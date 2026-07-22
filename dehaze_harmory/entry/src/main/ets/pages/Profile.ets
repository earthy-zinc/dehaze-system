import { router, promptAction } from '@kit.ArkUI';
import { AuthAPI, AuthUserInfo } from 'dehaze-sdk-harmony';
import { AuthState, AppKeys } from '../common/AppState';

/**
 * 个人中心页面
 *
 * 功能：
 * - 显示用户头像、昵称、用户名
 * - 菜单项：历史记录、我的收藏、设置、关于、退出登录
 * - 调用 AuthAPI.getMe() 获取用户信息
 * - 调用 AuthAPI.logout() 退出登录
 */
@Entry
@Component
struct Profile {
  @StorageLink(AppKeys.NICKNAME) nickname: string = '';
  @StorageLink(AppKeys.AVATAR) avatar: string = '';
  @StorageLink(AppKeys.IS_LOGGED_IN) isLoggedIn: boolean = false;

  @State username: string = '';
  @State roles: string = '';
  @State loading: boolean = false;
  @State loggingOut: boolean = false;

  aboutToAppear(): void {
    if (!this.isLoggedIn) {
      router.replaceUrl({ url: 'pages/Login' });
      return;
    }
    this.loadUserInfo();
  }

  /** 加载用户信息 */
  async loadUserInfo(): Promise<void> {
    this.loading = true;
    try {
      const userInfo: AuthUserInfo = await AuthAPI.getMe();
      this.username = userInfo.username || '';
      this.roles = (userInfo.roles || []).join(', ');

      // 更新 AppStorage
      AppStorage.setOrCreate<string>(AppKeys.NICKNAME, userInfo.nickname || userInfo.username);
      AppStorage.setOrCreate<string>(AppKeys.AVATAR, userInfo.avatar || '');
    } catch (e) {
      // 加载失败使用本地缓存数据
    } finally {
      this.loading = false;
    }
  }

  /** 退出登录 */
  async handleLogout(): Promise<void> {
    if (this.loggingOut) {
      return;
    }
    this.loggingOut = true;
    try {
      await AuthAPI.logout();
    } catch (e) {
      // 登出接口失败也清除本地状态
    }
    await AuthState.onLogout();
    promptAction.showToast({ message: '已退出登录' });
    setTimeout(() => {
      router.replaceUrl({ url: 'pages/Login' });
    }, 600);
  }

  /** 显示关于信息 */
  showAbout(): void {
    promptAction.showDialog({
      title: '关于',
      message: '图像去雾系统 v1.0.0\n\n基于深度学习的图像去雾处理系统，支持 29 种去雾算法。\n\n数据集：NH-HAZE-2023',
      buttons: [
        { text: '确定', color: '#389e3c' },
      ],
    });
  }

  /** 菜单项点击 */
  handleMenu(item: string): void {
    switch (item) {
      case 'history':
        router.pushUrl({ url: 'pages/ImageInput' });
        break;
      case 'favorites':
        router.pushUrl({ url: 'pages/AlgorithmSelect' });
        break;
      case 'settings':
        promptAction.showToast({ message: '设置功能开发中' });
        break;
      case 'about':
        this.showAbout();
        break;
      case 'logout':
        this.handleLogout();
        break;
      default:
        break;
    }
  }

  @Builder
  MenuItem(icon: string, title: string, desc: string, item: string, showArrow: boolean = true) {
    Row() {
      Text(icon)
        .fontSize(20)
        .width(32)

      Column() {
        Text(title)
          .fontSize(15)
          .fontColor('#333333')
        if (desc) {
          Text(desc)
            .fontSize(12)
            .fontColor('#999999')
            .margin({ top: 2 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)
      .margin({ left: 12 })

      if (showArrow) {
        Text('›')
          .fontSize(20)
          .fontColor('#CCCCCC')
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 14, bottom: 14 })
    .backgroundColor('#FFFFFF')
    .onClick(() => {
      this.handleMenu(item);
    })
  }

  build() {
    Column() {
      // 顶部栏
      Row() {
        Text('返回')
          .fontSize(15)
          .fontColor('#389e3c')
          .onClick(() => {
            router.back();
          })

        Text('个人中心')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333333')
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Text('').width(40)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor('#FFFFFF')

      Scroll() {
        Column() {
          // 用户信息卡片
          Column() {
            if (this.avatar) {
              Image(this.avatar)
                .width(80)
                .height(80)
                .borderRadius(40)
                .objectFit(ImageFit.Cover)
                .backgroundColor('#F0F0F0')
            } else {
              Text('👤')
                .fontSize(40)
                .width(80)
                .height(80)
                .textAlign(TextAlign.Center)
                .backgroundColor('#F0F9F0')
                .borderRadius(40)
            }

            Text(this.nickname || '用户')
              .fontSize(20)
              .fontWeight(FontWeight.Bold)
              .fontColor('#333333')
              .margin({ top: 12 })

            if (this.username) {
              Text(`@${this.username}`)
                .fontSize(13)
                .fontColor('#999999')
                .margin({ top: 4 })
            }

            if (this.roles) {
              Text(`角色: ${this.roles}`)
                .fontSize(12)
                .fontColor('#389e3c')
                .margin({ top: 4 })
                .padding({ left: 8, right: 8, top: 4, bottom: 4 })
                .backgroundColor('#F0F9F0')
                .borderRadius(8)
            }

            if (this.loading) {
              LoadingProgress()
                .width(24)
                .height(24)
                .color('#389e3c')
                .margin({ top: 8 })
            }
          }
          .width('100%')
          .padding({ top: 32, bottom: 32 })
          .backgroundColor('#FFFFFF')
          .alignItems(HorizontalAlign.Center)

          // 菜单组 1
          Column() {
            this.MenuItem('🕐', '历史记录', '查看处理历史', 'history')
          }
          .width('92%')
          .margin({ top: 16 })
          .borderRadius(12)
          .clip(true)
          .backgroundColor('#FFFFFF')

          // 菜单组 2
          Column() {
            this.MenuItem('⭐', '我的收藏', '收藏的算法', 'favorites')
            Divider().color('#F0F0F0').strokeWidth(1).margin({ left: 16, right: 16 })
            this.MenuItem('⚙️', '设置', '应用设置', 'settings')
          }
          .width('92%')
          .margin({ top: 12 })
          .borderRadius(12)
          .clip(true)
          .backgroundColor('#FFFFFF')

          // 菜单组 3
          Column() {
            this.MenuItem('ℹ️', '关于', '版本信息', 'about')
          }
          .width('92%')
          .margin({ top: 12 })
          .borderRadius(12)
          .clip(true)
          .backgroundColor('#FFFFFF')

          // 退出登录
          Column() {
            Row() {
              if (this.loggingOut) {
                LoadingProgress().width(20).height(20).color('#ff4d4f')
                Text('退出中...')
                  .fontSize(15)
                  .fontColor('#ff4d4f')
                  .margin({ left: 8 })
              } else {
                Text('🚪 退出登录')
                  .fontSize(15)
                  .fontColor('#ff4d4f')
              }
            }
            .width('100%')
            .padding({ top: 16, bottom: 16 })
            .justifyContent(FlexAlign.Center)
            .backgroundColor('#FFFFFF')
            .borderRadius(12)
            .onClick(() => {
              this.handleLogout();
            })
          }
          .width('92%')
          .margin({ top: 16, bottom: 24 })
        }
      }
      .layoutWeight(1)
      .scrollBar(BarState.Off)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F7FA')
  }
}
