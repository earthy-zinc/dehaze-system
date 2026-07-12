import { router, promptAction } from '@kit.ArkUI';
import { AuthAPI, AuthUserInfo } from 'dehaze-sdk-harmony';
import { AuthState, AppKeys } from '../common/AppState';

/**
 * 首页
 *
 * 功能入口导航，展示去雾系统核心功能模块：
 * - 图像输入：上传/拍照/样例/历史
 * - 算法管理：算法列表与选择
 * - 去雾处理：执行去雾
 * - 效果对比：对比处理前后效果
 * - 个人中心：用户信息与登出
 */
@Entry
@Component
struct Home {
  @StorageLink(AppKeys.NICKNAME) nickname: string = '';
  @StorageLink(AppKeys.IS_LOGGED_IN) isLoggedIn: boolean = false;

  aboutToAppear(): void {
    if (!this.isLoggedIn) {
      router.replaceUrl({ url: 'pages/Login' });
    }
  }

  /** 退出登录 */
  async handleLogout(): Promise<void> {
    try {
      await AuthAPI.logout();
    } catch (e) {
      // 登出接口失败也清除本地状态
    }
    await AuthState.onLogout();
    router.replaceUrl({ url: 'pages/Login' });
  }

  /** 功能项点击 */
  handleNavigate(page: string): void {
    router.pushUrl({ url: page });
  }

  @Builder
  FeatureCard(icon: string, title: string, desc: string, color: string, action: () => void) {
    Column() {
      Text(icon)
        .fontSize(32)
        .margin({ bottom: 12 })

      Text(title)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor('#333333')
        .margin({ bottom: 4 })

      Text(desc)
        .fontSize(12)
        .fontColor('#999999')
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
    }
    .width('100%')
    .height(140)
    .padding(16)
    .backgroundColor('#FFFFFF')
    .borderRadius(12)
    .alignItems(HorizontalAlign.Center)
    .justifyContent(FlexAlign.Center)
    .shadow({ radius: 8, color: 'rgba(0,0,0,0.04)', offsetX: 0, offsetY: 2 })
    .onClick(() => action())
  }

  build() {
    Column() {
      // 顶部栏
      Row() {
        Column() {
          Text(`你好，${this.nickname || '用户'}`)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
          Text('图像去雾系统')
            .fontSize(13)
            .fontColor('#999999')
            .margin({ top: 4 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        // 退出登录
        Text('退出')
          .fontSize(14)
          .fontColor('#999999')
          .padding({ left: 12, right: 12, top: 8, bottom: 8 })
          .backgroundColor('#F5F5F5')
          .borderRadius(20)
          .onClick(() => {
            this.handleLogout();
          })
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 16, bottom: 16 })

      // 功能区
      Scroll() {
        Column() {
          // 核心功能
          Text('核心功能')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .width('100%')
            .padding({ left: 20, top: 16, bottom: 12 })

          Grid() {
            GridItem() {
              this.FeatureCard('📷', '图像输入', '上传/拍照/样例图片', '#389e3c', () => {
                this.handleNavigate('pages/ImageInput');
              })
            }
            GridItem() {
              this.FeatureCard('⚡', '算法选择', '浏览与选择去雾算法', '#1890ff', () => {
                this.handleNavigate('pages/AlgorithmSelect');
              })
            }
            GridItem() {
              this.FeatureCard('🌫️', '去雾处理', '执行图像去雾处理', '#fa8c16', () => {
                this.handleNavigate('pages/Processing');
              })
            }
            GridItem() {
              this.FeatureCard('🔍', '效果对比', '对比处理前后效果', '#722ed1', () => {
                this.handleNavigate('pages/Compare');
              })
            }
          }
          .columnsTemplate('1fr 1fr')
          .columnsGap(12)
          .rowsGap(12)
          .width('92%')
          .height(300)

          // 使用流程
          Text('使用流程')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .width('100%')
            .padding({ left: 20, top: 24, bottom: 12 })

          Column() {
            Row() {
              Text('1').fontSize(14).fontColor('#389e3c').fontWeight(FontWeight.Bold)
              Text('上传或选择需要去雾的图片').fontSize(14).fontColor('#666666').margin({ left: 12 })
            }.width('100%').padding({ top: 8, bottom: 8 })

            Row() {
              Text('2').fontSize(14).fontColor('#389e3c').fontWeight(FontWeight.Bold)
              Text('选择合适的去雾算法').fontSize(14).fontColor('#666666').margin({ left: 12 })
            }.width('100%').padding({ top: 8, bottom: 8 })

            Row() {
              Text('3').fontSize(14).fontColor('#389e3c').fontWeight(FontWeight.Bold)
              Text('执行去雾处理，实时查看进度').fontSize(14).fontColor('#666666').margin({ left: 12 })
            }.width('100%').padding({ top: 8, bottom: 8 })

            Row() {
              Text('4').fontSize(14).fontColor('#389e3c').fontWeight(FontWeight.Bold)
              Text('对比效果并保存结果').fontSize(14).fontColor('#666666').margin({ left: 12 })
            }.width('100%').padding({ top: 8, bottom: 8 })
          }
          .width('92%')
          .padding(16)
          .backgroundColor('#FFFFFF')
          .borderRadius(12)
          .margin({ bottom: 24 })
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
