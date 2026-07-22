import { router, promptAction } from '@kit.ArkUI';
import { AlgorithmAPI, Algorithm } from 'dehaze-sdk-harmony';
import { ProcessingContext } from '../common/AppState';

/**
 * 算法选择页面
 *
 * 功能（对应算法选择模块需求规格）：
 * - 树形展示算法分类与算法
 * - 点击算法卡片选择
 * - 搜索筛选
 * - 收藏/取消收藏算法
 * 选中后跳转到去雾处理页
 */
@Entry
@Component
struct AlgorithmSelect {
  @State algorithmList: Algorithm[] = [];
  @State loading: boolean = false;
  @State keywords: string = '';
  @State expandedIds: Set<number> = new Set();
  // 收藏的算法 ID 列表
  @State favoriteIds: number[] = [];
  @State favoriteToggling: Set<number> = new Set();

  aboutToAppear(): void {
    this.loadAlgorithms();
    this.loadFavorites();
  }

  /** 加载算法列表 */
  async loadAlgorithms(): Promise<void> {
    this.loading = true;
    try {
      const result = await AlgorithmAPI.getList(this.keywords ? { keywords: this.keywords } : undefined);
      this.algorithmList = result || [];
    } catch (e) {
      promptAction.showToast({ message: '加载算法列表失败' });
    } finally {
      this.loading = false;
    }
  }

  /** 加载已收藏的算法列表 */
  async loadFavorites(): Promise<void> {
    try {
      const favorites = await AlgorithmAPI.listFavorites();
      this.favoriteIds = (favorites || []).map((a: Algorithm) => a.id);
    } catch (e) {
      // 加载收藏失败静默处理
    }
  }

  /** 切换展开/收起 */
  toggleExpand(id: number): void {
    if (this.expandedIds.has(id)) {
      this.expandedIds.delete(id);
    } else {
      this.expandedIds.add(id);
    }
    // 触发刷新
    this.expandedIds = new Set(this.expandedIds);
  }

  /** 切换收藏 */
  async toggleFavorite(id: number): Promise<void> {
    if (this.favoriteToggling.has(id)) {
      return;
    }
    this.favoriteToggling.add(id);
    this.favoriteToggling = new Set(this.favoriteToggling);

    const wasFavorite = this.favoriteIds.includes(id);
    try {
      await AlgorithmAPI.toggleFavorite(id);
      if (wasFavorite) {
        this.favoriteIds = this.favoriteIds.filter((fid: number) => fid !== id);
        promptAction.showToast({ message: '已取消收藏' });
      } else {
        this.favoriteIds = [...this.favoriteIds, id];
        promptAction.showToast({ message: '已收藏' });
      }
    } catch (e) {
      promptAction.showToast({ message: '操作失败，请重试' });
    } finally {
      this.favoriteToggling.delete(id);
      this.favoriteToggling = new Set(this.favoriteToggling);
    }
  }

  /** 判断是否已收藏 */
  isFavorite(id: number): boolean {
    return this.favoriteIds.includes(id);
  }

  /** 选择算法 */
  selectAlgorithm(algo: Algorithm): void {
    ProcessingContext.setAlgorithm(algo.id, algo.name);
    promptAction.showToast({ message: `已选择: ${algo.name}` });
    router.pushUrl({ url: 'pages/Processing' });
  }

  /** 递归构建树形节点 */
  @Builder
  AlgorithmNode(algo: Algorithm, depth: number) {
    Column() {
      Row() {
        // 缩进
        Blank().width(depth * 20)

        // 展开/收起图标（有子节点时）
        if (algo.children && algo.children.length > 0) {
          Text(this.expandedIds.has(algo.id) ? '▼' : '▶')
            .fontSize(12)
            .fontColor('#999999')
            .width(20)
        } else {
          Blank().width(20)
        }

        // 算法图标
        Text(algo.children && algo.children.length > 0 ? '📁' : '⚡')
          .fontSize(16)
          .margin({ left: 8, right: 8 })

        // 算法信息
        Column() {
          Text(algo.name)
            .fontSize(15)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333333')
          if (algo.description) {
            Text(algo.description)
              .fontSize(12)
              .fontColor('#999999')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .margin({ top: 2 })
          }
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        // 叶子节点：收藏按钮 + 选择按钮
        if (!algo.children || algo.children.length === 0) {
          // 收藏按钮
          Text(this.isFavorite(algo.id) ? '⭐' : '☆')
            .fontSize(18)
            .fontColor(this.isFavorite(algo.id) ? '#fa8c16' : '#CCCCCC')
            .width(32)
            .height(32)
            .textAlign(TextAlign.Center)
            .onClick(() => {
              this.toggleFavorite(algo.id);
            })

          // 选择按钮
          Text('选择')
            .fontSize(13)
            .fontColor('#389e3c')
            .padding({ left: 12, right: 12, top: 6, bottom: 6 })
            .border({ width: 1, color: '#389e3c' })
            .borderRadius(16)
            .margin({ left: 4 })
        }
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 12, bottom: 12 })
      .backgroundColor('#FFFFFF')
      .borderRadius(8)
      .onClick(() => {
        if (algo.children && algo.children.length > 0) {
          this.toggleExpand(algo.id);
        } else {
          this.selectAlgorithm(algo);
        }
      })

      // 子节点
      if (algo.children && algo.children.length > 0 && this.expandedIds.has(algo.id)) {
        ForEach(algo.children, (child: Algorithm) => {
          this.AlgorithmNode(child, depth + 1)
        })
      }
    }
    .width('100%')
    .margin({ top: 4 })
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

        Text('算法选择')
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

      // 搜索栏
      Row() {
        TextInput({ text: this.keywords, placeholder: '搜索算法名称' })
          .height(40)
          .fontSize(14)
          .borderRadius(20)
          .backgroundColor('#F5F5F5')
          .layoutWeight(1)
          .onChange((value: string) => {
            this.keywords = value;
          })

        Button('搜索')
          .height(40)
          .fontSize(14)
          .backgroundColor('#389e3c')
          .fontColor('#FFFFFF')
          .borderRadius(20)
          .margin({ left: 8 })
          .onClick(() => {
            this.loadAlgorithms();
          })
      }
      .width('92%')
      .margin({ top: 12, bottom: 12 })

      // 当前选择的原图
      if (ProcessingContext.getOriginalUrl()) {
        Row() {
          Image(ProcessingContext.getOriginalUrl())
            .width(40)
            .height(40)
            .borderRadius(4)
            .objectFit(ImageFit.Cover)

          Text('已选择原图，请选择算法')
            .fontSize(13)
            .fontColor('#666666')
            .margin({ left: 8 })
        }
        .width('92%')
        .padding(12)
        .backgroundColor('#F0F9F0')
        .borderRadius(8)
        .margin({ bottom: 8 })
      }

      // 算法列表
      if (this.loading) {
        Column() {
          LoadingProgress().width(48).height(48)
          Text('加载中...').fontSize(14).fontColor('#999999').margin({ top: 12 })
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else if (this.algorithmList.length === 0) {
        Column() {
          Text('📭').fontSize(48).margin({ bottom: 16 })
          Text('暂无算法数据').fontSize(14).fontColor('#999999')
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else {
        Scroll() {
          Column() {
            ForEach(this.algorithmList, (algo: Algorithm) => {
              this.AlgorithmNode(algo, 0)
            })
          }
          .width('92%')
          .padding({ bottom: 24 })
        }
        .layoutWeight(1)
        .scrollBar(BarState.Off)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F7FA')
  }
}
