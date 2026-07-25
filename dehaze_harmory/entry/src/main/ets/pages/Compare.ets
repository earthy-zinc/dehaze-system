import { router, promptAction } from '@kit.ArkUI';
import { EvaluationAPI, EvalParam, EvalResult, PollOptions } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/** 指标展示项（由 metrics Record 转换得到） */
class MetricItem {
  name: string = '';
  value: string = '';
}

/**
 * 效果对比页面
 *
 * 功能（对应效果对比模块需求规格 F-M05）：
 * - 并排对比模式：原图与处理结果上下排列
 * - 重叠对比模式：滑动分隔线对比
 * - 放大镜模式：跟随触摸位置放大显示
 * - 滤镜模式：灰度、暖色、冷色等滤镜效果对比
 * - 指标评估：调用 EvaluationAPI.evaluateAndWait 异步计算 PSNR/SSIM 等指标（GT 必须是 clean 图）
 * - 算法信息展示
 * - 重新处理/返回首页（结果由后端自动保存到 prediction_log）
 */
@Entry
@Component
struct Compare {
  @State originalUrl: string = '';
  @State resultUrl: string = '';
  @State algorithmName: string = '';
  @State algorithmId: number = 0;
  @State processingTime: number = 0;
  @State gtUrl: string = '';
  @State mode: number = 0; // 0=并排, 1=重叠, 2=放大镜, 3=滤镜
  @State sliderPosition: number = 50; // 重叠模式分隔线位置（0-100）

  // 放大镜模式
  @State magnifierX: number = 0;
  @State magnifierY: number = 0;
  @State magnifierVisible: boolean = false;

  // 滤镜模式（0=原图, 1=灰度, 2=暖色, 3=冷色, 4=复古）
  @State filterIndex: number = 0;

  // 指标评估
  @State evalMetrics: MetricItem[] = [];
  @State evalLoading: boolean = false;
  @State evalDone: boolean = false;
  @State evalError: string = '';

  aboutToAppear(): void {
    this.originalUrl = ProcessingContext.getOriginalUrl();
    this.resultUrl = ProcessingContext.getResultUrl();
    this.algorithmName = ProcessingContext.getAlgorithmName();
    this.algorithmId = ProcessingContext.getAlgorithmId();
    this.processingTime = ProcessingContext.getProcessingTime();
    this.gtUrl = ProcessingContext.getGtUrl();

    if (!this.originalUrl || !this.resultUrl) {
      promptAction.showToast({ message: '无对比数据' });
      setTimeout(() => {
        router.replaceUrl({ url: 'pages/Home' });
      }, 1000);
    }
  }

  /** 切换对比模式 */
  switchMode(mode: number): void {
    this.mode = mode;
    this.magnifierVisible = false;
  }

  /**
   * 保存结果
   * 后端在执行预测（PredictionAPI.predict）时会自动将结果写入 prediction_log，
   * 因此前端无需额外调用保存接口，仅做提示。
   */
  saveResult(): void {
    promptAction.showToast({ message: '结果已自动保存到历史记录' });
  }

  /** 执行指标评估（GT 必须是 clean 图） */
  async runEvaluation(): Promise<void> {
    if (!this.gtUrl) {
      promptAction.showToast({ message: '缺少 GT 参考图（clean），无法评估' });
      return;
    }
    if (!this.resultUrl) {
      promptAction.showToast({ message: '缺少结果图，无法评估' });
      return;
    }
    this.evalLoading = true;
    this.evalError = '';
    try {
      const param: EvalParam = {
        algorithmId: this.algorithmId,
        predUrl: this.resultUrl,
        gtUrl: this.gtUrl,
      };
      const pollOptions: PollOptions = {
        intervalMs: 2000,
        timeoutMs: 120000,
      };
      const result: EvalResult = await EvaluationAPI.evaluateAndWait(param, pollOptions);

      if (result.status === 'failed') {
        this.evalError = result.errorMessage || '评估失败';
        promptAction.showToast({ message: this.evalError });
        return;
      }

      const metrics: Record<string, number> = result.metrics || {};
      const items: MetricItem[] = [];
      Object.keys(metrics).forEach((key: string) => {
        const item = new MetricItem();
        item.name = key;
        item.value = metrics[key].toFixed(4);
        items.push(item);
      });
      this.evalMetrics = items;
      this.evalDone = true;
      promptAction.showToast({ message: '指标评估完成' });
    } catch (e) {
      const err = e as { message?: string };
      this.evalError = err?.message || '评估失败，请重试';
      promptAction.showToast({ message: this.evalError });
    } finally {
      this.evalLoading = false;
    }
  }

  /** 重新处理 */
  reprocess(): void {
    router.replaceUrl({ url: 'pages/AlgorithmSelect' });
  }

  /** 返回首页 */
  goHome(): void {
    router.replaceUrl({ url: 'pages/Home' });
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

        Text('效果对比')
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

      // 模式切换
      Scroll() {
        Row() {
          Text('并排对比')
            .fontSize(13)
            .fontWeight(this.mode === 0 ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.mode === 0 ? '#389e3c' : '#999999')
            .padding({ left: 12, right: 12, top: 8, bottom: 8 })
            .borderRadius(16)
            .backgroundColor(this.mode === 0 ? '#F0F9F0' : 'transparent')
            .onClick(() => {
              this.switchMode(0);
            })

          Text('重叠对比')
            .fontSize(13)
            .fontWeight(this.mode === 1 ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.mode === 1 ? '#389e3c' : '#999999')
            .padding({ left: 12, right: 12, top: 8, bottom: 8 })
            .borderRadius(16)
            .backgroundColor(this.mode === 1 ? '#F0F9F0' : 'transparent')
            .onClick(() => {
              this.switchMode(1);
            })

          Text('放大镜')
            .fontSize(13)
            .fontWeight(this.mode === 2 ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.mode === 2 ? '#389e3c' : '#999999')
            .padding({ left: 12, right: 12, top: 8, bottom: 8 })
            .borderRadius(16)
            .backgroundColor(this.mode === 2 ? '#F0F9F0' : 'transparent')
            .onClick(() => {
              this.switchMode(2);
            })

          Text('滤镜')
            .fontSize(13)
            .fontWeight(this.mode === 3 ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.mode === 3 ? '#389e3c' : '#999999')
            .padding({ left: 12, right: 12, top: 8, bottom: 8 })
            .borderRadius(16)
            .backgroundColor(this.mode === 3 ? '#F0F9F0' : 'transparent')
            .onClick(() => {
              this.switchMode(3);
            })
        }
        .padding({ left: 8, right: 8 })
      }
      .scrollable(ScrollDirection.Horizontal)
      .scrollBar(BarState.Off)
      .width('100%')
      .padding({ top: 8, bottom: 8 })
      .backgroundColor('#FFFFFF')
      .border({ width: { bottom: 1 }, color: '#EEEEEE' })

      // 对比区域
      Scroll() {
        Column() {
          if (this.mode === 0) {
            // 并排对比（上下排列）
            Column() {
              Text('原图')
                .fontSize(14)
                .fontColor('#999999')
                .margin({ bottom: 8 })

              Image(this.originalUrl)
                .width('90%')
                .height(220)
                .borderRadius(12)
                .objectFit(ImageFit.Contain)
                .backgroundColor('#F0F0F0')

              Text('去雾结果')
                .fontSize(14)
                .fontColor('#389e3c')
                .fontWeight(FontWeight.Medium)
                .margin({ top: 20, bottom: 8 })

              Image(this.resultUrl)
                .width('90%')
                .height(220)
                .borderRadius(12)
                .objectFit(ImageFit.Contain)
                .backgroundColor('#F0F0F0')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Center)
            .padding({ top: 16, bottom: 16 })
          } else if (this.mode === 1) {
            // 重叠对比（滑动分隔线）
            Column() {
              Stack() {
                // 结果图（底层）
                Image(this.resultUrl)
                  .width('90%')
                  .height(300)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)

                // 原图（上层，通过 clip 裁剪左侧）
                Column() {
                  Image(this.originalUrl)
                    .width('100%')
                    .height(300)
                    .objectFit(ImageFit.Cover)
                }
                .width(`${this.sliderPosition}%`)
                .height(300)
                .borderRadius(12)
                .clip(true)

                // 分隔线
                Row() {
                  Column() {
                    Text('◀')
                      .fontSize(16)
                      .fontColor('#FFFFFF')
                    Text('▶')
                      .fontSize(16)
                      .fontColor('#FFFFFF')
                  }
                  .width(32)
                  .height(64)
                  .backgroundColor('#389e3c')
                  .borderRadius(16)
                  .justifyContent(FlexAlign.Center)
                  .alignItems(HorizontalAlign.Center)
                }
                .width('100%')
                .height(300)
                .justifyContent(FlexAlign.Start)
                .padding({ left: `${this.sliderPosition}%` })
              }
              .width('100%')
              .alignItems(HorizontalAlign.Center)

              // 滑动条
              Slider({ value: this.sliderPosition, min: 0, max: 100, step: 1 })
                .width('85%')
                .blockColor('#389e3c')
                .selectedColor('#389e3c')
                .margin({ top: 16 })
                .onChange((value: number) => {
                  this.sliderPosition = value;
                })

              Row() {
                Text('原图').fontSize(12).fontColor('#999999')
                Blank().layoutWeight(1)
                Text('结果').fontSize(12).fontColor('#999999')
              }
              .width('85%')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Center)
            .padding({ top: 16, bottom: 16 })
          } else if (this.mode === 2) {
            // 放大镜模式
            Column() {
              Text('触摸图片以放大查看')
                .fontSize(13)
                .fontColor('#999999')
                .margin({ bottom: 12, top: 16 })

              Stack() {
                // 原图
                Image(this.originalUrl)
                  .width('90%')
                  .height(300)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')

                // 结果图（半透明叠加）
                Image(this.resultUrl)
                  .width('90%')
                  .height(300)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .opacity(0.5)

                // 放大镜（圆形区域放大结果图）
                if (this.magnifierVisible) {
                  Stack() {
                    Image(this.resultUrl)
                      .width(540)
                      .height(540)
                      .objectFit(ImageFit.Cover)
                      .position({
                        x: -this.magnifierX * 2.7,
                        y: -this.magnifierY * 2.7,
                      })
                  }
                  .width(120)
                  .height(120)
                  .borderRadius(60)
                  .clip(true)
                  .border({ width: 3, color: '#389e3c' })
                  .backgroundColor('#FFFFFF')
                  .position({
                    x: this.magnifierX - 60,
                    y: this.magnifierY - 60,
                  })
                }
              }
              .width('90%')
              .height(300)
              .borderRadius(12)
              .margin({ bottom: 16 })
              .onTouch((event: TouchEvent) => {
                const touch = event.touches[0];
                if (touch) {
                  this.magnifierX = touch.x;
                  this.magnifierY = touch.y;
                  if (event.type === TouchType.Down || event.type === TouchType.Move) {
                    this.magnifierVisible = true;
                  } else if (event.type === TouchType.Up || event.type === TouchType.Cancel) {
                    this.magnifierVisible = false;
                  }
                }
              })

              // 单独显示结果图
              Text('去雾结果')
                .fontSize(14)
                .fontColor('#389e3c')
                .fontWeight(FontWeight.Medium)
                .margin({ top: 16, bottom: 8 })

              Image(this.resultUrl)
                .width('90%')
                .height(220)
                .borderRadius(12)
                .objectFit(ImageFit.Contain)
                .backgroundColor('#F0F0F0')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Center)
            .padding({ top: 16, bottom: 16 })
          } else {
            // 滤镜模式
            Column() {
              // 滤镜选择
              Row() {
                Text('无').fontSize(12).fontColor(this.filterIndex === 0 ? '#389e3c' : '#999999')
                  .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                  .borderRadius(12)
                  .backgroundColor(this.filterIndex === 0 ? '#F0F9F0' : 'transparent')
                  .onClick(() => { this.filterIndex = 0; })
                Text('灰度').fontSize(12).fontColor(this.filterIndex === 1 ? '#389e3c' : '#999999')
                  .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                  .borderRadius(12)
                  .backgroundColor(this.filterIndex === 1 ? '#F0F9F0' : 'transparent')
                  .onClick(() => { this.filterIndex = 1; })
                Text('暖色').fontSize(12).fontColor(this.filterIndex === 2 ? '#389e3c' : '#999999')
                  .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                  .borderRadius(12)
                  .backgroundColor(this.filterIndex === 2 ? '#F0F9F0' : 'transparent')
                  .onClick(() => { this.filterIndex = 2; })
                Text('冷色').fontSize(12).fontColor(this.filterIndex === 3 ? '#389e3c' : '#999999')
                  .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                  .borderRadius(12)
                  .backgroundColor(this.filterIndex === 3 ? '#F0F9F0' : 'transparent')
                  .onClick(() => { this.filterIndex = 3; })
                Text('复古').fontSize(12).fontColor(this.filterIndex === 4 ? '#389e3c' : '#999999')
                  .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                  .borderRadius(12)
                  .backgroundColor(this.filterIndex === 4 ? '#F0F9F0' : 'transparent')
                  .onClick(() => { this.filterIndex = 4; })
              }
              .margin({ top: 16, bottom: 16 })

              Text('原图')
                .fontSize(13)
                .fontColor('#999999')
                .margin({ bottom: 6 })

              Image(this.originalUrl)
                .width('90%')
                .height(180)
                .borderRadius(12)
                .objectFit(ImageFit.Cover)
                .backgroundColor('#F0F0F0')

              Text('结果（含滤镜）')
                .fontSize(13)
                .fontColor('#389e3c')
                .fontWeight(FontWeight.Medium)
                .margin({ top: 16, bottom: 6 })

              if (this.filterIndex === 0) {
                Image(this.resultUrl)
                  .width('90%')
                  .height(180)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')
              } else if (this.filterIndex === 1) {
                Image(this.resultUrl)
                  .width('90%')
                  .height(180)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')
                  .grayscale(1)
              } else if (this.filterIndex === 2) {
                Image(this.resultUrl)
                  .width('90%')
                  .height(180)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')
                  .hueRotate(-20)
                  .saturate(1.3)
              } else if (this.filterIndex === 3) {
                Image(this.resultUrl)
                  .width('90%')
                  .height(180)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')
                  .hueRotate(20)
                  .saturate(0.8)
              } else if (this.filterIndex === 4) {
                Image(this.resultUrl)
                  .width('90%')
                  .height(180)
                  .borderRadius(12)
                  .objectFit(ImageFit.Cover)
                  .backgroundColor('#F0F0F0')
                  .sepia(0.6)
                  .contrast(1.1)
              }
            }
            .width('100%')
            .alignItems(HorizontalAlign.Center)
            .padding({ top: 16, bottom: 16 })
          }

          // 算法信息
          Column() {
            Text('处理信息')
              .fontSize(15)
              .fontWeight(FontWeight.Bold)
              .fontColor('#333333')
              .margin({ bottom: 12 })

            Row() {
              Text('算法').fontSize(13).fontColor('#999999').width(80)
              Text(this.algorithmName).fontSize(13).fontColor('#333333')
            }.width('100%').padding({ top: 6, bottom: 6 })

            Row() {
              Text('耗时').fontSize(13).fontColor('#999999').width(80)
              Text(this.processingTime > 0 ? `${this.processingTime} ms` : '-').fontSize(13).fontColor('#333333')
            }.width('100%').padding({ top: 6, bottom: 6 })

            Row() {
              Text('GT 参考').fontSize(13).fontColor('#999999').width(80)
              Text(this.gtUrl ? '✅ 已加载 clean 图' : '❌ 无 GT').fontSize(13)
                .fontColor(this.gtUrl ? '#389e3c' : '#ff4d4f')
            }.width('100%').padding({ top: 6, bottom: 6 })

            Row() {
              Text('状态').fontSize(13).fontColor('#999999').width(80)
              Text('✅ 处理成功').fontSize(13).fontColor('#389e3c')
            }.width('100%').padding({ top: 6, bottom: 6 })
          }
          .width('90%')
          .padding(16)
          .backgroundColor('#FFFFFF')
          .borderRadius(12)
          .margin({ bottom: 16 })

          // 指标评估模块
          Column() {
            Row() {
              Text('指标评估')
                .fontSize(15)
                .fontWeight(FontWeight.Bold)
                .fontColor('#333333')
                .layoutWeight(1)

              if (this.evalLoading) {
                LoadingProgress().width(20).height(20).color('#389e3c')
              } else if (!this.evalDone) {
                Text('开始评估')
                  .fontSize(13)
                  .fontColor('#389e3c')
                  .padding({ left: 12, right: 12, top: 6, bottom: 6 })
                  .border({ width: 1, color: '#389e3c' })
                  .borderRadius(16)
                  .onClick(() => {
                    this.runEvaluation();
                  })
              }
            }
            .width('100%')
            .margin({ bottom: 12 })

            if (this.evalDone && this.evalMetrics.length > 0) {
              ForEach(this.evalMetrics, (item: MetricItem) => {
                Row() {
                  Text(item.name).fontSize(13).fontColor('#666666').width(80)
                  Text(item.value).fontSize(13).fontColor('#333333').fontWeight(FontWeight.Medium).layoutWeight(1)
                }.width('100%').padding({ top: 6, bottom: 6 })
              })
            } else if (this.evalDone && this.evalMetrics.length === 0) {
              Text('暂无评估结果数据').fontSize(13).fontColor('#999999')
            } else if (this.evalError) {
              Text(this.evalError).fontSize(12).fontColor('#ff4d4f')
            } else if (!this.gtUrl) {
              Text('需要 GT 参考图（clean 图）才能进行指标评估').fontSize(12).fontColor('#ff4d4f')
            } else {
              Text('点击"开始评估"计算 PSNR/SSIM 等指标').fontSize(12).fontColor('#999999')
            }
          }
          .width('90%')
          .padding(16)
          .backgroundColor('#FFFFFF')
          .borderRadius(12)
          .margin({ bottom: 16 })

          // 操作按钮
          Row() {
            Button('保存结果')
              .layoutWeight(1)
              .height(44)
              .fontSize(14)
              .backgroundColor('#389e3c')
              .fontColor('#FFFFFF')
              .borderRadius(8)
              .margin({ right: 8 })
              .onClick(() => {
                this.saveResult();
              })

            Button('重新处理')
              .layoutWeight(1)
              .height(44)
              .fontSize(14)
              .backgroundColor('#FFFFFF')
              .fontColor('#389e3c')
              .border({ width: 1, color: '#389e3c' })
              .borderRadius(8)
              .margin({ right: 8 })
              .onClick(() => {
                this.reprocess();
              })

            Button('首页')
              .layoutWeight(1)
              .height(44)
              .fontSize(14)
              .backgroundColor('#FFFFFF')
              .fontColor('#666666')
              .border({ width: 1, color: '#DDDDDD' })
              .borderRadius(8)
              .onClick(() => {
                this.goHome();
              })
          }
          .width('90%')
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
