import { router, promptAction } from '@kit.ArkUI';
import { ProcessingContext } from '../common/AppState';

/**
 * 效果对比页面
 *
 * 功能（对应效果对比模块需求规格 F-M05）：
 * - 并排对比模式：原图与处理结果上下排列
 * - 重叠对比模式：滑动分隔线对比
 * - 算法信息展示
 * - 保存结果/重新处理/返回首页
 */
@Entry
@Component
struct Compare {
  @State originalUrl: string = '';
  @State resultUrl: string = '';
  @State algorithmName: string = '';
  @State processingTime: number = 0;
  @State mode: number = 0; // 0=并排, 1=重叠
  @State sliderPosition: number = 50; // 重叠模式分隔线位置（0-100）

  aboutToAppear(): void {
    this.originalUrl = ProcessingContext.getOriginalUrl();
    this.resultUrl = ProcessingContext.getResultUrl();
    this.algorithmName = ProcessingContext.getAlgorithmName();
    this.processingTime = ProcessingContext.getProcessingTime();

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
  }

  /** 保存结果（提示） */
  saveResult(): void {
    promptAction.showToast({ message: '结果已保存到历史记录' });
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
      Row() {
        Text('并排对比')
          .fontSize(14)
          .fontWeight(this.mode === 0 ? FontWeight.Bold : FontWeight.Normal)
          .fontColor(this.mode === 0 ? '#389e3c' : '#999999')
          .padding({ left: 16, right: 16, top: 8, bottom: 8 })
          .borderRadius(16)
          .backgroundColor(this.mode === 0 ? '#F0F9F0' : 'transparent')
          .onClick(() => {
            this.switchMode(0);
          })

        Text('重叠对比')
          .fontSize(14)
          .fontWeight(this.mode === 1 ? FontWeight.Bold : FontWeight.Normal)
          .fontColor(this.mode === 1 ? '#389e3c' : '#999999')
          .padding({ left: 16, right: 16, top: 8, bottom: 8 })
          .borderRadius(16)
          .backgroundColor(this.mode === 1 ? '#F0F9F0' : 'transparent')
          .onClick(() => {
            this.switchMode(1);
          })
      }
      .width('100%')
      .justifyContent(FlexAlign.Center)
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
          } else {
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
              Text('状态').fontSize(13).fontColor('#999999').width(80)
              Text('✅ 处理成功').fontSize(13).fontColor('#389e3c')
            }.width('100%').padding({ top: 6, bottom: 6 })
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
