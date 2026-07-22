import { router } from '@kit.ArkUI';
import { promptAction } from '@kit.ArkUI';
import { AlgorithmAPI, DatasetAPI, Algorithm, DatasetItem } from 'dehaze-sdk-harmony';
import { AppKeys, ProcessingContext } from '../common/AppState';

/**
 * 首页
 *
 * 功能入口导航，展示去雾系统核心功能模块：
 * - 顶部：用户问候 + 个人中心入口
 * - 核心功能：图像输入/算法选择/去雾处理/效果对比
 * - 算法推荐：展示推荐算法
 * - 效果展示：NH-HAZE-2023 hazy/clean 对比图
 * - 使用流程引导
 */
@Entry
@Component
struct Home {
  @StorageLink(AppKeys.NICKNAME) nickname: string = '';
  @StorageLink(AppKeys.AVATAR) avatar: string = '';
  @StorageLink(AppKeys.IS_LOGGED_IN) isLoggedIn: boolean = false;

  // 推荐算法
  @State recommendAlgorithms: Algorithm[] = [];
  // 效果展示（hazy/clean 对比）
  @State showcaseItems: DatasetItem[] = [];
  @State showcaseLoading: boolean = false;

  aboutToAppear(): void {
    if (!this.isLoggedIn) {
      router.replaceUrl({ url: 'pages/Login' });
      return;
    }
    this.loadRecommendAlgorithms();
    this.loadShowcaseItems();
  }

  /** 加载推荐算法（取前 4 个） */
  async loadRecommendAlgorithms(): Promise<void> {
    try {
      const list = await AlgorithmAPI.getList(undefined);
      // 展平树形结构，取叶子节点（实际算法）
      const flatList: Algorithm[] = [];
      const flatten = (items: Algorithm[]): void => {
        for (const item of items) {
          if (item.children && item.children.length > 0) {
            flatten(item.children);
          } else {
            flatList.push(item);
          }
        }
      };
      flatten(list || []);
      this.recommendAlgorithms = flatList.slice(0, 4);
    } catch (e) {
      // 加载失败静默处理
    }
  }

  /** 加载效果展示样例（NH-HAZE-2023 hazy/clean 对比） */
  async loadShowcaseItems(): Promise<void> {
    this.showcaseLoading = true;
    try {
      const result = await DatasetAPI.getDatasetItems({
        pageNum: 1,
        pageSize: 6,
      });
      this.showcaseItems = result.list || [];
    } catch (e) {
      // 加载失败静默处理
    } finally {
      this.showcaseLoading = false;
    }
  }

  /** 从样例中提取 hazy 和 clean URL */
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
      if (!hazyUrl && item.imageUrls.length > 0) {
        hazyUrl = item.imageUrls[0].url;
      }
    }
    return { hazyUrl, cleanUrl };
  }

  /** 功能项点击 */
  handleNavigate(page: string): void {
    router.pushUrl({ url: page });
  }

  /** 跳转到个人中心 */
  handleProfile(): void {
    router.pushUrl({ url: 'pages/Profile' });
  }

  /** 选择推荐算法 */
  selectRecommendAlgorithm(algo: Algorithm): void {
    ProcessingContext.setAlgorithm(algo.id, algo.name);
    promptAction.showToast({ message: `已选择: ${algo.name}` });
    router.pushUrl({ url: 'pages/ImageInput' });
  }

  /** 查看效果展示样例 */
  viewShowcaseItem(item: DatasetItem): void {
    const { hazyUrl, cleanUrl } = this.extractHazyAndClean(item);
    if (!hazyUrl) {
      promptAction.showToast({ message: '该样例无图片' });
      return;
    }
    ProcessingContext.setOriginal(hazyUrl, '', cleanUrl);
    router.pushUrl({ url: 'pages/AlgorithmSelect' });
  }

  @Builder
  FeatureCard(icon: string, title: string, desc: string, action: () => void) {
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

        // 个人中心入口
        Row() {
          if (this.avatar) {
            Image(this.avatar)
              .width(40)
              .height(40)
              .borderRadius(20)
              .objectFit(ImageFit.Cover)
              .backgroundColor('#F0F0F0')
          } else {
            Text('👤')
              .fontSize(22)
              .width(40)
              .height(40)
              .textAlign(TextAlign.Center)
              .backgroundColor('#F0F9F0')
              .borderRadius(20)
          }
        }
        .onClick(() => {
          this.handleProfile();
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
              this.FeatureCard('📷', '图像输入', '上传/拍照/样例图片', () => {
                this.handleNavigate('pages/ImageInput');
              })
            }
            GridItem() {
              this.FeatureCard('⚡', '算法选择', '浏览与选择去雾算法', () => {
                this.handleNavigate('pages/AlgorithmSelect');
              })
            }
            GridItem() {
              this.FeatureCard('🌫️', '去雾处理', '执行图像去雾处理', () => {
                this.handleNavigate('pages/Processing');
              })
            }
            GridItem() {
              this.FeatureCard('🔍', '效果对比', '对比处理前后效果', () => {
                this.handleNavigate('pages/Compare');
              })
            }
          }
          .columnsTemplate('1fr 1fr')
          .columnsGap(12)
          .rowsGap(12)
          .width('92%')
          .height(300)

          // 算法推荐
          Text('算法推荐')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .width('100%')
            .padding({ left: 20, top: 24, bottom: 12 })

          if (this.recommendAlgorithms.length === 0) {
            Text('暂无推荐算法')
              .fontSize(13)
              .fontColor('#999999')
              .width('92%')
              .padding(16)
              .textAlign(TextAlign.Center)
          } else {
            Scroll() {
              Row() {
                ForEach(this.recommendAlgorithms, (algo: Algorithm) => {
                  Column() {
                    Text('⚡')
                      .fontSize(28)
                      .margin({ bottom: 8 })
                    Text(algo.name)
                      .fontSize(13)
                      .fontWeight(FontWeight.Medium)
                      .fontColor('#333333')
                      .maxLines(1)
                      .textOverflow({ overflow: TextOverflow.Ellipsis })
                    if (algo.description) {
                      Text(algo.description)
                        .fontSize(11)
                        .fontColor('#999999')
                        .maxLines(2)
                        .textOverflow({ overflow: TextOverflow.Ellipsis })
                        .margin({ top: 4 })
                        .textAlign(TextAlign.Center)
                    }
                  }
                  .width(140)
                  .padding(12)
                  .backgroundColor('#FFFFFF')
                  .borderRadius(12)
                  .margin({ right: 12 })
                  .alignItems(HorizontalAlign.Center)
                  .onClick(() => {
                    this.selectRecommendAlgorithm(algo);
                  })
                })
              }
              .padding({ left: 20, right: 20 })
            }
            .scrollable(ScrollDirection.Horizontal)
            .scrollBar(BarState.Off)
            .width('100%')
          }

          // 效果展示
          Text('效果展示')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .width('100%')
            .padding({ left: 20, top: 24, bottom: 12 })

          if (this.showcaseLoading) {
            Row() {
              LoadingProgress().width(32).height(32).color('#389e3c')
              Text('加载样例中...').fontSize(13).fontColor('#999999').margin({ left: 8 })
            }
            .width('92%')
            .padding(16)
            .justifyContent(FlexAlign.Center)
          } else if (this.showcaseItems.length === 0) {
            Text('暂无效果展示样例')
              .fontSize(13)
              .fontColor('#999999')
              .width('92%')
              .padding(16)
              .textAlign(TextAlign.Center)
          } else {
            Column() {
              ForEach(this.showcaseItems, (item: DatasetItem) => {
                Row() {
                  // hazy 图
                  Column() {
                    Text('有雾').fontSize(11).fontColor('#999999').margin({ bottom: 4 })
                    Image(this.extractHazyAndClean(item).hazyUrl)
                      .width(120)
                      .height(90)
                      .borderRadius(8)
                      .objectFit(ImageFit.Cover)
                      .backgroundColor('#F0F0F0')
                  }

                  Text('→').fontSize(20).fontColor('#389e3c').margin({ left: 12, right: 12 })

                  // clean 图
                  Column() {
                    Text('清晰').fontSize(11).fontColor('#389e3c').margin({ bottom: 4 })
                    Image(this.extractHazyAndClean(item).cleanUrl)
                      .width(120)
                      .height(90)
                      .borderRadius(8)
                      .objectFit(ImageFit.Cover)
                      .backgroundColor('#F0F0F0')
                  }

                  Column() {
                    Text('查看').fontSize(12).fontColor('#389e3c')
                  }
                  .margin({ left: 12 })
                }
                .width('100%')
                .padding(12)
                .backgroundColor('#FFFFFF')
                .borderRadius(12)
                .margin({ bottom: 8 })
                .onClick(() => {
                  this.viewShowcaseItem(item);
                })
              })
            }
            .width('92%')
          }

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
              Text('执行去雾处理，实时查看用时').fontSize(14).fontColor('#666666').margin({ left: 12 })
            }.width('100%').padding({ top: 8, bottom: 8 })

            Row() {
              Text('4').fontSize(14).fontColor('#389e3c').fontWeight(FontWeight.Bold)
              Text('对比效果并评估指标').fontSize(14).fontColor('#666666').margin({ left: 12 })
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
