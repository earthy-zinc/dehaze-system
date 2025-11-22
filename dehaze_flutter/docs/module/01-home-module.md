# 首页模块设计文档

**文档版本**: v1.0
**最后更新**: 2025-11-22
**关联文档**: [模块概览](./README.md) | [架构设计](../architecture/00-overview.md) | [设计系统](../design/01-design-system.md)

---

## 📋 模块概述

### 功能定位

首页模块是Flutter图像去雾系统的用户入口，承担着产品介绍、功能引导、快速体验和用户教育的重要职责。该模块设计旨在为用户提供直观友好的第一印象，帮助用户快速了解产品价值并开始使用核心功能。

### 核心价值

- **用户引导**: 清晰的功能介绍和使用引导，降低学习成本
- **快速体验**: 提供一键式样例体验，让用户快速感受产品效果
- **价值展示**: 通过效果展示和数据统计，建立用户对产品的信任
- **便捷导航**: 提供快捷的导航入口，提升用户操作效率

### 模块边界

| 输入来源 | 处理内容 | 输出结果 | 依赖模块 |
|---------|---------|---------|---------|
| **应用启动** | 初始化界面、加载数据 | 欢迎页面、功能引导 | 各功能模块 |
| **用户操作** | 页面导航、功能选择 | 页面跳转、状态更新 | 算法选择、图像输入 |
| **系统数据** | 统计信息、更新内容 | 展示内容、功能提示 | 各功能模块 |

---

## 🏗️ 架构设计

### Clean Architecture分层

```
features/home/
├── data/                              # 数据层
│   ├── datasources/                   # 数据源
│   │   ├── home_datasource.dart            # 首页数据源
│   │   ├── showcase_datasource.dart        # 展示数据源
│   │   └── stats_datasource.dart           # 统计数据源
│   ├── models/                         # 数据模型
│   │   ├── showcase_item_model.dart        # 展示项模型
│   │   ├── home_stats_model.dart           # 统计数据模型
│   │   └── feature_card_model.dart         # 功能卡片模型
│   └── repositories/                   # 仓储实现
│       └── home_repository_impl.dart
├── domain/                            # 领域层
│   ├── entities/                      # 业务实体
│   │   ├── showcase_item.dart               # 展示项实体
│   │   ├── home_statistics.dart             # 首页统计实体
│   │   ├── feature_highlight.dart           # 功能亮点实体
│   │   └── user_guide_item.dart             // 用户指南项
│   ├── repositories/                  # 仓储接口
│   │   └── home_repository.dart
│   └── usecases/                       # 用例
│       ├── load_home_data_usecase.dart      # 加载首页数据
│       ├── get_showcase_items_usecase.dart  # 获取展示项
│       ├── get_statistics_usecase.dart      # 获取统计数据
│       └── track_user_action_usecase.dart   # 跟踪用户行为
└── presentation/                      # 表现层
    ├── pages/                         # 页面组件
    │   ├── home_page.dart                   # 首页主页面
    │   ├── welcome_page.dart                // 欢迎引导页
    │   └── onboarding_page.dart             // 新手引导页
    ├── widgets/                       # 可复用组件
    │   ├── hero_section_widget.dart         # Hero区域组件
    │   ├── feature_grid_widget.dart         # 功能网格组件
    │   ├── showcase_carousel_widget.dart    # 展示轮播组件
    │   ├── stats_section_widget.dart        # 统计区域组件
    │   ├── quick_action_widget.dart         // 快捷操作组件
    │   └── testimonial_widget.dart          // 用户评价组件
    └── providers/                      # 状态管理
        └── home_provider.dart                 # 首页状态管理
```

### 领域实体设计

```dart
/// 首页展示项
class ShowcaseItem {
  final String id;
  final String title;
  final String description;
  final String beforeImageUrl;
  final String afterImageUrl;
  final String algorithm;
  final String category;
  final Map<String, dynamic> metadata;

  const ShowcaseItem({
    required this.id,
    required this.title,
    required this.description,
    required this.beforeImageUrl,
    required this.afterImageUrl,
    required this.algorithm,
    required this.category,
    this.metadata = const {},
  });
}

/// 首页统计数据
class HomeStatistics {
  final int totalProcessed;
  final int activeAlgorithms;
  final int supportedFormats;
  final double averageProcessingTime;
  final int userCount;
  final double satisfactionScore;

  const HomeStatistics({
    required this.totalProcessed,
    required this.activeAlgorithms,
    required this.supportedFormats,
    required this.averageProcessingTime,
    required this.userCount,
    required this.satisfactionScore,
  });
}
```

---

## 🎨 界面设计

### 首页布局结构

```
┌─────────────────────────────────────────────────────────────┐
│  图像去雾系统                                    [设置] [帮助] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🌟 Hero Section - 产品核心价值展示                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                                                        │ │
│  │    🌫️ 图像去雾                                         │ │
│  │    专业级图像处理系统                                   │ │
│  │                                                        │ │
│  │    采用先进的深度学习算法，一键还原清晰视界              │ │
│  │    从图像输入到效果评估的完整闭环体验                    │ │
│  │                                                        │ │
│  │    [立即开始] → [浏览数据集]                             │ │
│  │                                                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  🎯 效果展示区 - 去雾前后对比                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            [去雾前] → [去雾后] 对比展示                    │ │
│  │                                                        │ │
│  │     [轮播展示多个处理效果对比]                           │ │
│  │                                                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ⚡ 快速体验区                                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  💡 使用样例图片快速体验去雾效果                           │ │
│  │                                                        │ │
│  │  [轻度雾霾] [中度雾霾] [重度雾霾] [夜景去雾] [更多样例]     │ │
│  │                                                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  🔧 核心功能介绍                                            │
│                                                             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ 📷 图像输入  │ 🧠 算法选择  │ ⚙️ 去雾处理  │ 📊 效果对比  │ │
│  │ 多种输入方式  │ 智能算法推荐  │ 实时进度展示  │ 多维对比评估  │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
│                                                             │
│  📊 系统统计                                                │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  总处理量: 1.2M+  |  活跃算法: 8+  |  平均耗时: 2.5s     │ │
│  │  用户数量: 50K+   |  支持格式: 12  |  满意度: 4.8/5     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  🎮 底部导航栏                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  🏠 首页  📷 输入  🧠 算法  ⚙️ 处理  📊 对比              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 状态管理

### Riverpod状态设计

```dart
/// 首页状态
@freezed
class HomeState with _$HomeState {
  const factory HomeState.initial() = _HomeInitial;
  const factory HomeState.loading() = _HomeLoading;
  const factory HomeState.loaded({
    required List<ShowcaseItem> showcaseItems,
    required HomeStatistics statistics,
    required List<FeatureHighlight> features,
  }) = _HomeLoaded;
  const factory HomeState.error(String message) = _HomeError;
}

/// 首页状态Provider
final homeStateProvider = StateNotifierProvider<HomeNotifier, HomeState>((ref) {
  return HomeNotifier(ref.read(homeRepositoryProvider));
});

/// 首页状态管理器
class HomeNotifier extends StateNotifier<HomeState> {
  final HomeRepository _repository;

  HomeNotifier(this._repository) : super(const HomeState.initial());

  /// 加载首页数据
  Future<void> loadHomeData() async {
    state = const HomeState.loading();
    try {
      final showcaseItems = await _repository.getShowcaseItems();
      final statistics = await _repository.getStatistics();
      final features = await _repository.getFeatureHighlights();

      state = HomeState.loaded(
        showcaseItems: showcaseItems,
        statistics: statistics,
        features: features,
      );
    } catch (e) {
      state = HomeState.error('加载首页数据失败: ${e.toString()}');
    }
  }
}

/// 仓储Provider
final homeRepositoryProvider = Provider<HomeRepository>((ref) {
  return HomeRepositoryImpl(ref.read(homeDatasourceProvider));
});
```

---

## 📈 性能监控

### 关键性能指标

| 指标名称 | 目标值 | 监控方法 |
|---------|--------|---------|
| **首页加载时间** | < 1s | 性能监控 |
| **Hero区域渲染** | < 300ms | UI性能监控 |
| **轮播切换延迟** | < 200ms | 交互性能监控 |
| **图片加载时间** | < 500ms | 图片加载监控 |
| **用户转化率** | > 60% | 用户行为分析 |

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**维护团队**: Flutter开发团队