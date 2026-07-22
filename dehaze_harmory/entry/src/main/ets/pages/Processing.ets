import { router, promptAction } from '@kit.ArkUI';
import { PredictionAPI, PredParam, PredResult } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/**
 * 去雾处理页面
 *
 * 功能（对应去雾处理模块需求规格 F-M04-001）：
 * - 显示原图和选中算法
 * - 算法参数调节（去雾强度等通用参数）
 * - 确认对话框后执行去雾
 * - 处理中显示 indeterminate loading + 真实用时计时器，不显示虚假进度百分比
 * - 支持取消处理
 * - 处理完成后跳转效果对比页
 */
@Entry
@Component
struct Processing {
  @State originalUrl: string = '';
  @State algorithmId: number = 0;
  @State algorithmName: string = '';

  @State isProcessing: boolean = false;
  @State showConfirm: boolean = false;
  @State resultUrl: string = '';

  // 真实用时计时（毫秒）
  @State elapsedMs: number = 0;
  private startTime: number = 0;
  private timerId: number = -1;
  // 取消标志：取消后忽略 Promise 结果
  private isCancelled: boolean = false;

  // 算法参数（通用）
  @State dehazeStrength: number = 50; // 去雾强度 0-100
  @State enhanceEnabled: boolean = false; // 是否开启细节增强

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

  aboutToDisappear(): void {
    this.stopTimer();
  }

  /** 显示确认对话框 */
  showConfirmDialog(): void {
    this.showConfirm = true;
  }

  /** 启动真实用时计时器 */
  private startTimer(): void {
    this.startTime = Date.now();
    this.elapsedMs = 0;
    this.timerId = setInterval(() => {
      this.elapsedMs = Date.now() - this.startTime;
    }, 100);
  }

  /** 停止计时器 */
  private stopTimer(): void {
    if (this.timerId !== -1) {
      clearInterval(this.timerId);
      this.timerId = -1;
    }
  }

  /** 构造算法参数 JSON */
  private buildParams(): string {
    const params: Record<string, number | boolean> = {
      dehazeStrength: this.dehazeStrength / 100, // 归一化到 0-1
      enhance: this.enhanceEnabled,
    };
    return JSON.stringify(params);
  }

  /** 执行去雾处理 */
  async startProcessing(): Promise<void> {
    this.showConfirm = false;
    this.isProcessing = true;
    this.isCancelled = false;
    this.startTimer();

    // 同步参数到上下文
    ProcessingContext.setAlgorithmParams(this.buildParams());

    try {
      // 调用预测接口（API 同步返回，无真实进度）
      const param: PredParam = {
        algorithmId: this.algorithmId,
        imageUrl: this.originalUrl,
        params: this.buildParams(),
      };
      const result: PredResult = await PredictionAPI.predict(param);

      // 取消后忽略结果
      if (this.isCancelled) {
        return;
      }

      this.stopTimer();

      // 保存结果到上下文
      ProcessingContext.setResult(result.resultUrl, result.time);

      promptAction.showToast({ message: '去雾处理完成' });

      // 跳转效果对比页
      setTimeout(() => {
        router.replaceUrl({ url: 'pages/Compare' });
      }, 600);
    } catch (e) {
      this.stopTimer();
      if (this.isCancelled) {
        return;
      }
      const err = e as { message?: string };
      promptAction.showToast({ message: err?.message || '处理失败，请重试' });
      this.isProcessing = false;
    }
  }

  /** 取消处理 */
  cancelProcessing(): void {
    this.isCancelled = true;
    this.stopTimer();
    this.isProcessing = false;
    promptAction.showToast({ message: '已取消处理' });
    router.back();
  }

  /** 格式化用时 */
  formatElapsed(): string {
    const seconds = Math.floor(this.elapsedMs / 1000);
    const ms = this.elapsedMs % 1000;
    return `${seconds}.${Math.floor(ms / 100)}s`;
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
          Text(`参数: 强度${this.dehazeStrength}%${this.enhanceEnabled ? ' + 细节增强' : ''}`)
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

  @Builder
  ParamAdjuster() {
    Column() {
      Text('参数配置')
        .fontSize(15)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
        .margin({ bottom: 12 })

      // 去雾强度
      Row() {
        Text('去雾强度').fontSize(14).fontColor('#666666').width(80)
        Slider({ value: this.dehazeStrength, min: 0, max: 100, step: 5 })
          .layoutWeight(1)
          .blockColor('#389e3c')
          .selectedColor('#389e3c')
        Text(`${this.dehazeStrength}%`).fontSize(13).fontColor('#389e3c').width(48).textAlign(TextAlign.End)
      }
      .width('100%')
      .padding({ top: 8, bottom: 8 })

      // 细节增强开关
      Row() {
        Text('细节增强').fontSize(14).fontColor('#666666').layoutWeight(1)
        Toggle({ type: ToggleType.Switch, isOn: this.enhanceEnabled })
          .selectedColor('#389e3c')
          .onChange((on: boolean) => {
            this.enhanceEnabled = on;
          })
      }
      .width('100%')
      .padding({ top: 8, bottom: 8 })
    }
    .width('80%')
    .padding(16)
    .backgroundColor('#FFFFFF')
    .borderRadius(12)
    .margin({ top: 16 })
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
          // 处理中：indeterminate loading + 真实用时 + 取消按钮
          Column() {
            // 原图预览
            Image(this.originalUrl)
              .width('70%')
              .height(200)
              .borderRadius(12)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#F0F0F0')

            // indeterminate loading
            LoadingProgress()
              .width(64)
              .height(64)
              .color('#389e3c')
              .margin({ top: 32 })

            // 处理中文案
            Text('正在执行去雾处理...')
              .fontSize(15)
              .fontColor('#333333')
              .margin({ top: 12 })

            // 真实用时
            Text(`已用时 ${this.formatElapsed()}`)
              .fontSize(13)
              .fontColor('#999999')
              .margin({ top: 6 })

            Text('处理为同步请求，完成后自动跳转')
              .fontSize(12)
              .fontColor('#BBBBBB')
              .margin({ top: 4 })

            // 取消按钮
            Button('取消处理')
              .width('60%')
              .height(44)
              .fontSize(15)
              .backgroundColor('#FFFFFF')
              .fontColor('#ff4d4f')
              .border({ width: 1, color: '#ff4d4f' })
              .borderRadius(8)
              .margin({ top: 40 })
              .onClick(() => {
                this.cancelProcessing();
              })
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
              .height(220)
              .borderRadius(12)
              .objectFit(ImageFit.Contain)
              .backgroundColor('#F0F0F0')

            // 算法信息
            Column() {
              Row() {
                Text('使用算法').fontSize(14).fontColor('#999999')
                Text(this.algorithmName).fontSize(14).fontColor('#333333').margin({ left: 12 })
              }.width('100%').padding({ top: 8, bottom: 8 })
            }
            .width('80%')
            .padding(16)
            .backgroundColor('#FFFFFF')
            .borderRadius(12)
            .margin({ top: 16 })

            // 参数调节
            this.ParamAdjuster()

            // 开始按钮
            Button('开始去雾')
              .width('80%')
              .height(48)
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .backgroundColor('#389e3c')
              .fontColor('#FFFFFF')
              .borderRadius(8)
              .margin({ top: 24 })
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
