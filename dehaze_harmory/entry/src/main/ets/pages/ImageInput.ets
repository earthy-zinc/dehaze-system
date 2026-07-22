import { router, promptAction, XComponentController } from '@kit.ArkUI';
import { photoAccessHelper } from '@kit.PhotoAccessHelper';
import { camera } from '@kit.CameraKit';
import { image } from '@kit.ImageKit';
import { fileio } from '@kit.CoreFileKit';
import { FileAPI, DatasetAPI, InputHistoryAPI, FileInfo, DatasetItem, InputHistory } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/**
 * 图像输入页面
 *
 * 功能（对应图像输入模块前端实现设计）：
 * - 上传：从相册选择图片上传
 * - 拍照：调用鸿蒙相机 API 拍照输入
 * - 样例：浏览公开数据项作为样例（区分 hazy/clean，hazy 作为输入，clean 作为 GT）
 * - 历史：查看历史处理记录
 * 选中图片后跳转到算法选择页
 */
@Entry
@Component
struct ImageInput {
  @State currentTab: number = 0;
  @State uploading: boolean = false;

  // 样例列表
  @State sampleList: DatasetItem[] = [];
  @State sampleLoading: boolean = false;

  // 历史记录
  @State historyList: InputHistory[] = [];
  @State historyLoading: boolean = false;

  // 相机相关
  @State surfaceId: string = '';
  @State cameraReady: boolean = false;
  @State cameraError: string = '';
  @State capturing: boolean = false;
  private cameraManager: camera.CameraManager | null = null;
  private session: camera.CaptureSession | null = null;
  private photoOutput: camera.PhotoOutput | null = null;
  private cameraInput: camera.CameraInput | null = null;
  private previewOutput: camera.PreviewOutput | null = null;
  private imageReceiver: image.ImageReceiver | null = null;
  private xComponentController: XComponentController = new XComponentController();

  aboutToAppear(): void {
    this.loadSamples();
    this.loadHistory();
  }

  aboutToDisappear(): void {
    this.releaseCamera();
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

      // 获取选中图片的 URI
      const asset = result[0];
      const uri = asset.uri;

      // 上传文件
      const fileInfo: FileInfo = await FileAPI.upload(uri);

      promptAction.showToast({ message: '上传成功' });

      // 保存到处理上下文（用户上传无对应 clean GT）
      ProcessingContext.setOriginal(fileInfo.url, uri, '');

      // 跳转到算法选择页
      router.pushUrl({ url: 'pages/AlgorithmSelect' });
    } catch (e) {
      const err = e as { message?: string };
      promptAction.showToast({ message: err?.message || '上传失败，请重试' });
    } finally {
      this.uploading = false;
    }
  }

  /** 初始化相机 */
  async initCamera(surfaceId: string): Promise<void> {
    if (this.cameraReady || this.cameraManager) {
      return;
    }
    try {
      this.cameraManager = camera.getCameraManager(getContext(this));
      const cameras = this.cameraManager.getSupportedCameras();
      if (cameras.length === 0) {
        this.cameraError = '未找到相机设备';
        return;
      }

      this.cameraInput = this.cameraManager.createCameraInput(cameras[0]);
      await this.cameraInput.open();

      const capability = this.cameraManager.getSupportedOutputCapability(cameras[0], camera.SceneMode.NORMAL);
      const previewProfile = capability.previewProfiles[0];
      const photoProfile = capability.photoProfiles[0];

      this.previewOutput = this.cameraManager.createPreviewOutput(previewProfile, surfaceId);

      this.imageReceiver = image.createImageReceiver(photoProfile.size.width, photoProfile.size.height, image.ImageFormat.JPEG, 8);
      const photoSurfaceId = await this.imageReceiver.getReceivingSurfaceId();
      this.photoOutput = this.cameraManager.createPhotoOutput(photoProfile, photoSurfaceId);

      this.session = this.cameraManager.createCaptureSession();
      this.session.beginConfig();
      this.session.addInput(this.cameraInput);
      this.session.addOutput(this.previewOutput);
      this.session.addOutput(this.photoOutput);
      await this.session.commitConfig();
      await this.session.start();

      this.cameraReady = true;
      this.cameraError = '';
    } catch (e) {
      const err = e as { message?: string };
      this.cameraError = `相机初始化失败: ${err?.message || '未知错误'}`;
    }
  }

  /** 释放相机资源 */
  async releaseCamera(): Promise<void> {
    try {
      if (this.session) {
        await this.session.stop();
        await this.session.release();
        this.session = null;
      }
      if (this.previewOutput) {
        await this.previewOutput.release();
        this.previewOutput = null;
      }
      if (this.photoOutput) {
        await this.photoOutput.release();
        this.photoOutput = null;
      }
      if (this.cameraInput) {
        await this.cameraInput.close();
        await this.cameraInput.release();
        this.cameraInput = null;
      }
      if (this.imageReceiver) {
        await this.imageReceiver.release();
        this.imageReceiver = null;
      }
      this.cameraManager = null;
      this.cameraReady = false;
    } catch (e) {
      // 释放失败静默处理
    }
  }

  /** 拍照并上传 */
  async takePhoto(): Promise<void> {
    if (!this.cameraReady || !this.photoOutput || !this.imageReceiver) {
      promptAction.showToast({ message: '相机未就绪' });
      return;
    }
    this.capturing = true;

    try {
      this.imageReceiver.on('imageArrival', async () => {
        try {
          const img = await this.imageReceiver!.readLatestImage();
          if (!img) {
            this.capturing = false;
            return;
          }
          const component = await img.getComponent(image.ComponentType.JPEG);
          if (!component || !component.byteBuffer) {
            this.capturing = false;
            return;
          }

          // 保存到缓存文件
          const context = getContext(this);
          const tempPath = `${context.cacheDir}/camera_${Date.now()}.jpg`;
          const fd = fileio.openSync(tempPath, fileio.OpenFlag.CREATE | fileio.OpenFlag.READ_WRITE);
          fileio.writeSync(fd, component.byteBuffer);
          fileio.closeSync(fd);
          img.release();

          // 上传
          this.uploading = true;
          const fileInfo: FileInfo = await FileAPI.upload(tempPath);
          promptAction.showToast({ message: '拍照上传成功' });
          ProcessingContext.setOriginal(fileInfo.url, tempPath, '');
          this.capturing = false;
          this.uploading = false;
          router.pushUrl({ url: 'pages/AlgorithmSelect' });
        } catch (e) {
          const err = e as { message?: string };
          promptAction.showToast({ message: err?.message || '拍照处理失败' });
          this.capturing = false;
          this.uploading = false;
        }
      });

      await this.photoOutput.capture();
    } catch (e) {
      const err = e as { message?: string };
      promptAction.showToast({ message: err?.message || '拍照失败' });
      this.capturing = false;
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

  /** 从样例中提取 hazy 图 URL（作为输入）和 clean 图 URL（作为 GT） */
  private extractHazyAndClean(item: DatasetItem): { hazyUrl: string; cleanUrl: string } {
    let hazyUrl = '';
    let cleanUrl = '';
    if (item.imageUrls) {
      for (const img of item.imageUrls) {
        if (img.type === 'hazy' && !hazyUrl) {
          hazyUrl = img.url;
        } else if (img.type === 'clear' && !cleanUrl) {
          cleanUrl = img.url;
        }
      }
      // 兜底：若未显式标记类型，取第一张作为 hazy 输入
      if (!hazyUrl && item.imageUrls.length > 0) {
        hazyUrl = item.imageUrls[0].url;
      }
    }
    return { hazyUrl, cleanUrl };
  }

  /** 选择样例图片 */
  async selectSample(item: DatasetItem): Promise<void> {
    const { hazyUrl, cleanUrl } = this.extractHazyAndClean(item);
    if (!hazyUrl) {
      promptAction.showToast({ message: '该样例无 hazy 图片' });
      return;
    }
    // 设置 hazy 为输入，clean 作为 GT 参考图
    ProcessingContext.setOriginal(hazyUrl, '', cleanUrl);
    promptAction.showToast({ message: cleanUrl ? '已选择样例图片' : '已选择样例（无对应 GT）' });
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
      ProcessingContext.setOriginal(item.originalImageUrl, '', '');
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
        if (index === 1 && !this.cameraReady && !this.surfaceId) {
          // 进入相机 tab 时延迟初始化
          setTimeout(() => this.initCamera(this.surfaceId), 200);
        }
        if (index !== 1 && this.cameraReady) {
          // 离开相机 tab 时释放
          this.releaseCamera();
        }
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
        this.TabBar('拍照', 1)
        this.TabBar('样例', 2)
        this.TabBar('历史', 3)
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
              LoadingProgress().width(48).height(48).color('#389e3c')
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
        // 拍照面板
        Stack() {
          if (this.cameraError) {
            Column() {
              Text('⚠️').fontSize(48).margin({ bottom: 16 })
              Text(this.cameraError)
                .fontSize(14)
                .fontColor('#ff4d4f')
                .textAlign(TextAlign.Center)
                .padding({ left: 24, right: 24 })
            }
            .width('100%')
            .height('100%')
            .justifyContent(FlexAlign.Center)
            .alignItems(HorizontalAlign.Center)
          } else {
            XComponent({
              id: 'cameraPreview',
              type: XComponentType.SURFACE,
              controller: this.xComponentController,
            })
              .onLoad(() => {
                // XComponent 加载后获取 surfaceId
                this.surfaceId = this.xComponentController.getXComponentSurfaceId();
                if (this.surfaceId) {
                  this.initCamera(this.surfaceId);
                }
              })
              .width('100%')
              .height('100%')

            // 顶部提示
            Text(this.cameraReady ? '相机已就绪' : '正在初始化相机...')
              .position({ top: 16 })
              .fontSize(13)
              .fontColor('#FFFFFF')
              .backgroundColor('rgba(0,0,0,0.5)')
              .padding({ left: 12, right: 12, top: 6, bottom: 6 })
              .borderRadius(16)
              .width('auto')

            // 拍照按钮
            Column() {
              Button(this.capturing ? '拍摄中...' : '📸 拍照')
                .width(140)
                .height(56)
                .fontSize(18)
                .backgroundColor('#389e3c')
                .fontColor('#FFFFFF')
                .borderRadius(28)
                .enabled(this.cameraReady && !this.capturing && !this.uploading)
                .onClick(() => {
                  this.takePhoto();
                })

              if (this.uploading) {
                Text('正在上传...')
                  .fontSize(12)
                  .fontColor('#FFFFFF')
                  .margin({ top: 8 })
              }
            }
            .width('100%')
            .height('100%')
            .justifyContent(FlexAlign.End)
            .alignItems(HorizontalAlign.Center)
            .padding({ bottom: 48 })
          }
        }
        .width('100%')
        .height('100%')
        .backgroundColor('#000000')
      } else if (this.currentTab === 2) {
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
                  // 缩略图（优先 hazy）
                  Image(this.extractHazyAndClean(item).hazyUrl)
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
                    Row() {
                      Text('hazy → clean').fontSize(11).fontColor('#389e3c')
                    }.margin({ top: 4 })
                    Text(this.extractHazyAndClean(item).cleanUrl ? '含 GT 参考图（clean）' : '无 GT 参考图')
                      .fontSize(11)
                      .fontColor('#999999')
                      .margin({ top: 2 })
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
