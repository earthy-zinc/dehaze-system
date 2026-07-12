import { router, promptAction } from '@kit.ArkUI';
import { PredictionAPI, PredParam, PredResult } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/**
 * 去雾处理页面
 *
 * 功能（对应去雾处理模块需求规格 F-M04-001）：
 * - 显示原图和选中算法
 * - 确认对话框后执行去雾
 * - 实时进度展示（预处理→算法初始化→去雾→后处理→保存）
 * - 处理完成后跳转效果对比页
 */
@Entry
@Component
struct Processing {
  @State originalUrl: string = '';
  @State algorithmId: number = 0;
  @State algorithmName: string = '';

  @State isProcessing: boolean = false;
  @State progress: number = 0;
  @State stage: string = '';
  @State showConfirm: boolean = false;
  @State resultUrl: string = '';

  aboutToAppear(): void {
    this.originalUrl = ProcessingContext.getOriginalUrl();
    this.algorithmId = ProcessingContext.getAlgorithmId();
    this.algorithmName = ProcessingContext.getAlgorithmName();

    // 未选择图片或算法时返回
    if (!this.originalUrl || !this.algorithmId) {
      promptAction.showToast({ message: '请先选择图片和算法' });
      setTimeout(() => {
        router.back();
      }, 1000);
    }
  }

  /** 显示确认对话框 */
  showConfirmDialog(): void {
    this.showConfirm = true;
  }

  /** 执行去雾处理 */
  async startProcessing(): Promise<void> {
    this.showConfirm = false;
    this.isProcessing = true;
    this.progress = 0;
    this.stage = '图片预处理...';

    try {
      // 模拟进度更新（实际处理是单次请求）
      const progressTimer = setInterval(() => {
        if (this.progress < 90) {
          this.progress += Math.random() * 15;
          if (this.progress < 20) {
            this.stage = '图片预处理...';
          } else if (this.progress < 40) {
            this.stage = '算法初始化...';
          } else if (this.progress < 80) {
            this.stage = '执行去雾算法...';
          } else {
            this.stage = '后处理优化...';
          }
        }
      }, 500);

      // 调用预测接口
      const param: PredParam = {
        algorithmId: this.algorithmId,
        imageUrl: this.originalUrl,
      };
      const result: PredResult = await PredictionAPI.predict(param);

      clearInterval(progressTimer);
      this.progress = 100;
      this.stage = '处理完成';

      // 保存结果到上下文
      ProcessingContext.setResult(result.resultUrl, result.time);

      promptAction.showToast({ message: '去雾处理完成' });

      // 跳转效果对比页
      setTimeout(() => {
        router.replaceUrl({ url: 'pages/Compare' });
      }, 800);
    } catch (e) {
      const err = e as { message?: string };
      promptAction.showToast({ message: err?.message || '处理失败，请重试' });
      this.isProcessing = false;
    }
  }

  @Builder
  ConfirmDialog() {
    Column() {
      Text('确认开始去雾')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
        .margin({ bottom: 16 })

      // 图片信息
      Row() {
        Image(this.originalUrl)
          .width(80)
          .height(80)
          .borderRadius(8)
          .objectFit(ImageFit.Cover)

        Column() {
          Text('算法信息')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333333')
          Text(`算法: ${this.algorithmName}`)
            .fontSize(13)
            .fontColor('#666666')
            .margin({ top: 4 })
          Text('参数: 默认配置')
            .fontSize(13)
            .fontColor('#999999')
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .margin({ left: 12 })
        .layoutWeight(1)
      }
      .width('100%')
      .margin({ bottom: 24 })

      // 按钮
      Row() {
        Button('取消')
          .layoutWeight(1)
          .height(44)
          .fontSize(16)
          .backgroundColor('#F5F5F5')
          .fontColor('#666666')
          .borderRadius(8)
          .margin({ right: 8 })
          .onClick(() => {
            this.showConfirm = false;
          })

        Button('开始处理')
          .layoutWeight(1)
          .height(44)
          .fontSize(16)
          .backgroundColor('#389e3c')
          .fontColor('#FFFFFF')
          .borderRadius(8)
          .onClick(() => {
            this.startProcessing();
          })
      }
      .width('100%')
    }
    .width('80%')
    .padding(24)
    .backgroundColor('#FFFFFF')
    .borderRadius(16)
  }

  build() {
    Stack() {
      Column() {
        // 顶部栏
        Row() {
          Text('返回')
            .fontSize(15)
            .fontColor('#389e3c')
            .onClick(() => {
              router.back();
            })

          Text('去雾处理')
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

        if (this.isProcessing) {
          // 处理中
          Column() {
            // 原图预览
            Image(this.originalUrl)
              .width('70%')
              .height(200)
              .borderRadius(12)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#F0F0F0')

            // 进度条
            Progress({ value: this.progress, total: 100, type: ProgressType.Linear })
              .width('80%')
              .color('#389e3c')
              .margin({ top: 32 })

            // 阶段文字
            Text(this.stage)
              .fontSize(15)
              .fontColor('#333333')
              .margin({ top: 12 })

            Text(`${Math.floor(this.progress)}%`)
              .fontSize(28)
              .fontWeight(FontWeight.Bold)
              .fontColor('#389e3c')
              .margin({ top: 8 })
          }
          .width('100%')
          .height('100%')
          .justifyContent(FlexAlign.Center)
          .alignItems(HorizontalAlign.Center)
        } else {
          // 待处理
          Column() {
            // 原图预览
            Text('原图')
              .fontSize(14)
              .fontColor('#999999')
              .margin({ bottom: 8 })

            Image(this.originalUrl)
              .width('80%')
              .height(240)
              .borderRadius(12)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#F0F0F0')

            // 算法信息
            Column() {
              Row() {
                Text('使用算法').fontSize(14).fontColor('#999999')
                Text(this.algorithmName).fontSize(14).fontColor('#333333').margin({ left: 12 })
              }.width('100%').padding({ top: 8, bottom: 8 })

              Row() {
                Text('参数配置').fontSize(14).fontColor('#999999')
                Text('默认配置').fontSize(14).fontColor('#333333').margin({ left: 12 })
              }.width('100%').padding({ top: 8, bottom: 8 })
            }
            .width('80%')
            .padding(16)
            .backgroundColor('#FFFFFF')
            .borderRadius(12)
            .margin({ top: 24 })

            // 开始按钮
            Button('开始去雾')
              .width('80%')
              .height(48)
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .backgroundColor('#389e3c')
              .fontColor('#FFFFFF')
              .borderRadius(8)
              .margin({ top: 32 })
              .onClick(() => {
                this.showConfirmDialog();
              })
          }
          .width('100%')
          .height('100%')
          .justifyContent(FlexAlign.Center)
          .alignItems(HorizontalAlign.Center)
        }
      }
      .width('100%')
      .height('100%')

      // 确认对话框遮罩
      if (this.showConfirm) {
        Column() {
          this.ConfirmDialog()
        }
        .width('100%')
        .height('100%')
        .backgroundColor('rgba(0,0,0,0.4)')
        .justifyContent(FlexAlign.Center)
        .alignItems(HorizontalAlign.Center)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F7FA')
  }
}
