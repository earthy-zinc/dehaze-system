import { router, promptAction } from '@kit.ArkUI';
import { photoAccessHelper } from '@kit.PhotoAccessHelper';
import { FileAPI, DatasetAPI, InputHistoryAPI, FileInfo, DatasetItem, InputHistory } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/**
 * 图像输入页面
 *
 * 功能（对应图像输入模块前端实现设计）：
 * - 上传：从相册选择图片上传
 * - 样例：浏览公开数据项作为样例
 * - 历史：查看历史处理记录
 * 选中图片后跳转到算法选择页
 */
@Entry
@Component
struct ImageInput {
  @State currentTab: number = 0;
  @State uploading: boolean = false;
  @State uploadProgress: number = 0;

  // 样例列表
  @State sampleList: DatasetItem[] = [];
  @State sampleLoading: boolean = false;

  // 历史记录
  @State historyList: InputHistory[] = [];
  @State historyLoading: boolean = false;

  aboutToAppear(): void {
    this.loadSamples();
    this.loadHistory();
  }

  /** 从相册选择图片并上传 */
  async pickAndUploadImage(): Promise<void> {
    try {
      const helper = photoAccessHelper.getPhotoAccessHelper(this.getUIContext());
      const result = await helper.selectAssets({
        maxSelectedNumber: 1,
        typeFilter: ['IMAGE'],
      });

      if (!result || result.length === 0) {
        return;
      }

      this.uploading = true;
      this.uploadProgress = 0;

      // 获取选中图片的 URI
      const asset = result[0];
      const uri = asset.uri;

      // 上传文件
      const fileInfo: FileInfo = await FileAPI.upload(uri);

      promptAction.showToast({ message: '上传成功' });

      // 保存到处理上下文
      ProcessingContext.setOriginal(fileInfo.url, uri);

      // 跳转到算法选择页
      router.pushUrl({ url: 'pages/AlgorithmSelect' });
    } catch (e) {
      const err = e as { message?: string };
      promptAction.showToast({ message: err?.message || '上传失败，请重试' });
    } finally {
      this.uploading = false;
    }
  }

  /** 加载样例图片（公开数据项） */
  async loadSamples(): Promise<void> {
    this.sampleLoading = true;
    try {
      const result = await DatasetAPI.getDatasetItems({
        pageNum: 1,
        pageSize: 20,
      });
      this.sampleList = result.list || [];
    } catch (e) {
      // 加载失败静默处理
    } finally {
      this.sampleLoading = false;
    }
  }

  /** 选择样例图片 */
  async selectSample(item: DatasetItem): Promise<void> {
    if (!item.imageUrls || item.imageUrls.length === 0) {
      promptAction.showToast({ message: '该样例无图片' });
      return;
    }
    const imageUrl = item.imageUrls[0].url;
    ProcessingContext.setOriginal(imageUrl);
    promptAction.showToast({ message: '已选择样例图片' });
    router.pushUrl({ url: 'pages/AlgorithmSelect' });
  }

  /** 加载历史记录 */
  async loadHistory(): Promise<void> {
    this.historyLoading = true;
    try {
      const result = await InputHistoryAPI.list({ pageNum: 1, pageSize: 20 });
      this.historyList = result.list || [];
    } catch (e) {
      // 加载失败静默处理
    } finally {
      this.historyLoading = false;
    }
  }

  /** 重新处理历史记录 */
  reprocessHistory(item: InputHistory): void {
    if (item.originalImageUrl) {
      ProcessingContext.setOriginal(item.originalImageUrl);
      if (item.algorithmId) {
        ProcessingContext.setAlgorithm(item.algorithmId, item.algorithmName || '');
        router.pushUrl({ url: 'pages/Processing' });
      } else {
        router.pushUrl({ url: 'pages/AlgorithmSelect' });
      }
    }
  }

  /** Tab 配置 */
  @Builder
  TabBar(title: string, index: number) {
    Text(title)
      .fontSize(15)
      .fontWeight(this.currentTab === index ? FontWeight.Bold : FontWeight.Normal)
      .fontColor(this.currentTab === index ? '#389e3c' : '#999999')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .onClick(() => {
        this.currentTab = index;
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

        Text('图像输入')
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

      // Tab 栏
      Row() {
        this.TabBar('上传', 0)
        this.TabBar('样例', 1)
        this.TabBar('历史', 2)
      }
      .width('100%')
      .justifyContent(FlexAlign.SpaceAround)
      .backgroundColor('#FFFFFF')
      .border({ width: { bottom: 1 }, color: '#EEEEEE' })

      // 内容区
      if (this.currentTab === 0) {
        // 上传面板
        Column() {
          if (this.uploading) {
            Column() {
              Progress({ value: this.uploadProgress, total: 100, type: ProgressType.Linear })
                .width('80%')
                .color('#389e3c')
              Text('上传中...')
                .fontSize(14)
                .fontColor('#999999')
                .margin({ top: 12 })
            }
            .width('100%')
            .margin({ top: 120 })
            .alignItems(HorizontalAlign.Center)
          } else {
            // 上传区域
            Column() {
              Text('📁')
                .fontSize(48)
                .margin({ bottom: 16 })

              Text('点击选择图片')
                .fontSize(16)
                .fontColor('#666666')
                .margin({ bottom: 8 })

              Text('支持 JPG / PNG / WEBP 格式')
                .fontSize(12)
                .fontColor('#999999')
            }
            .width('85%')
            .height(240)
            .margin({ top: 60 })
            .borderRadius(12)
            .border({ width: 2, color: '#389e3c', style: BorderStyle.Dashed })
            .backgroundColor('#F0F9F0')
            .justifyContent(FlexAlign.Center)
            .alignItems(HorizontalAlign.Center)
            .onClick(() => {
              this.pickAndUploadImage();
            })

            // 说明
            Text('上传后自动进入算法选择')
              .fontSize(13)
              .fontColor('#999999')
              .margin({ top: 24 })
          }
        }
        .width('100%')
        .height('100%')
        .alignItems(HorizontalAlign.Center)
      } else if (this.currentTab === 1) {
        // 样例图片库
        if (this.sampleLoading) {
          Column() {
            LoadingProgress().width(48).height(48)
            Text('加载中...').fontSize(14).fontColor('#999999').margin({ top: 12 })
          }.width('100%').margin({ top: 120 }).alignItems(HorizontalAlign.Center)
        } else if (this.sampleList.length === 0) {
          Column() {
            Text('📭').fontSize(48).margin({ bottom: 16 })
            Text('暂无样例图片').fontSize(14).fontColor('#999999')
          }.width('100%').margin({ top: 120 }).alignItems(HorizontalAlign.Center)
        } else {
          List({ space: 12 }) {
            ForEach(this.sampleList, (item: DatasetItem) => {
              ListItem() {
                Row() {
                  // 缩略图
                  Image(item.imageUrls && item.imageUrls.length > 0 ? item.imageUrls[0].url : '')
                    .width(80)
                    .height(80)
                    .borderRadius(8)
                    .backgroundColor('#F0F0F0')
                    .objectFit(ImageFit.Cover)

                  Column() {
                    Text(`样例 ${item.id}`)
                      .fontSize(15)
                      .fontWeight(FontWeight.Medium)
                      .fontColor('#333333')
                    Text(item.imageUrls && item.imageUrls.length > 0 ? item.imageUrls[0].type : '')
                      .fontSize(12)
                      .fontColor('#999999')
                      .margin({ top: 4 })
                  }
                  .alignItems(HorizontalAlign.Start)
                  .margin({ left: 12 })
                  .layoutWeight(1)

                  Text('使用')
                    .fontSize(13)
                    .fontColor('#389e3c')
                    .padding({ left: 12, right: 12, top: 6, bottom: 6 })
                    .border({ width: 1, color: '#389e3c' })
                    .borderRadius(16)
                }
                .width('100%')
                .padding(12)
                .backgroundColor('#FFFFFF')
                .borderRadius(12)
              }
              .onClick(() => {
                this.selectSample(item);
              })
            })
          }
          .width('92%')
          .margin({ top: 12 })
          .layoutWeight(1)
        }
      } else {
        // 历史记录
        if (this.historyLoading) {
          Column() {
            LoadingProgress().width(48).height(48)
            Text('加载中...').fontSize(14).fontColor('#999999').margin({ top: 12 })
          }.width('100%').margin({ top: 120 }).alignItems(HorizontalAlign.Center)
        } else if (this.historyList.length === 0) {
          Column() {
            Text('🕐').fontSize(48).margin({ bottom: 16 })
            Text('暂无历史记录').fontSize(14).fontColor('#999999')
          }.width('100%').margin({ top: 120 }).alignItems(HorizontalAlign.Center)
        } else {
          List({ space: 12 }) {
            ForEach(this.historyList, (item: InputHistory) => {
              ListItem() {
                Row() {
                  // 缩略图
                  Image(item.originalThumbnailUrl || item.originalImageUrl || '')
                    .width(64)
                    .height(64)
                    .borderRadius(8)
                    .backgroundColor('#F0F0F0')
                    .objectFit(ImageFit.Cover)

                  Column() {
                    Text(item.algorithmName || '未知算法')
                      .fontSize(14)
                      .fontWeight(FontWeight.Medium)
                      .fontColor('#333333')
                    Text(item.createTime || '')
                      .fontSize(12)
                      .fontColor('#999999')
                      .margin({ top: 4 })
                    Row() {
                      Text(item.status === 1 ? '✅ 成功' : item.status === 2 ? '❌ 失败' : '🔄 处理中')
                        .fontSize(11)
                        .fontColor(item.status === 1 ? '#389e3c' : item.status === 2 ? '#ff4d4f' : '#fa8c16')
                    }.margin({ top: 4 })
                  }
                  .alignItems(HorizontalAlign.Start)
                  .margin({ left: 12 })
                  .layoutWeight(1)

                  Text('重处理')
                    .fontSize(13)
                    .fontColor('#389e3c')
                    .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                    .border({ width: 1, color: '#389e3c' })
                    .borderRadius(16)
                }
                .width('100%')
                .padding(12)
                .backgroundColor('#FFFFFF')
                .borderRadius(12)
              }
              .onClick(() => {
                this.reprocessHistory(item);
              })
            })
          }
          .width('92%')
          .margin({ top: 12 })
          .layoutWeight(1)
        }
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F7FA')
  }
}
