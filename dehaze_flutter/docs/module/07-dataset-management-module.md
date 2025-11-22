# 数据集管理模块设计文档

**文档版本**: v1.0
**最后更新**: 2025-11-22
**关联文档**: [模块概览](./README.md) | [架构设计](../architecture/02-architecture.md) | [设计系统](../design/01-design-system.md)

---

## 📋 模块概述

### 功能定位

数据集管理模块负责管理系统中的图像数据集，提供浏览、搜索、分类、批量处理等功能。该模块面向需要大量处理图像的专业用户和研究人员，提供便捷的数据集管理和批量处理能力。

### 核心价值

- **高效浏览**: 瀑布流布局支持大量图像的快速浏览
- **智能分类**: 基于AI的图像自动分类和标签功能
- **批量处理**: 支持多图像的批量去雾处理
- **数据管理**: 完整的数据集导入、导出和管理功能

### 模块边界

| 输入来源 | 处理内容 | 输出结果 | 依赖模块 |
|---------|---------|---------|---------|
| **用户操作** | 数据浏览、筛选、批量选择 | 选中的图像集 | 图像输入模块 |
| **文件导入** | 数据集上传、解析、分类 | 数据集元数据 | 算法选择模块 |
| **批量处理** | 多图像任务创建 | 批量处理任务 | 去雾处理模块 |

---

## 🏗️ 架构设计

### 领域实体设计

```dart
/// 数据集实体
class Dataset {
  final String id;
  final String name;
  final String description;
  final DatasetType type;
  final int imageCount;
  final int totalSize;
  final List<String> categories;
  final Map<String, int> distribution;
  final DateTime createdAt;

  const Dataset({
    required this.id,
    required this.name,
    required this.description,
    required this.type,
    required this.imageCount,
    required this.totalSize,
    required this.categories,
    required this.distribution,
    required this.createdAt,
  });
}

enum DatasetType {
  sample,    // 样例数据集
  test,      // 测试数据集
  user,      // 用户数据集
  result,    // 处理结果集
}
```

---

## 🎨 界面设计

### 数据集页面布局

```
┌─────────────────────────────────────────────────────────────┐
│  数据集管理                                  [上传] [创建] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔍 搜索和筛选                                               │
│  [搜索框...] [类型筛选] [分类筛选] [排序方式▼]               │
│                                                             │
│  📊 数据集统计                                               │
│  样例集: 5个 | 测试集: 3个 | 用户集: 12个 | 总计: 10,235张   │
│                                                             │
│  🖼️ 瀑布流展示                                               │
│                                                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │图片1│ │图片2│ │图片3│ │图片4│ │图片5│ │图片6│ ...        │
│  │120×80│ │200×150│ │180×120│ │160×100│ │200×160│ │140×90│     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │
│                                                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │图片7│ │图片8│ │图片9│ │图片10│ │图片11│ │图片12│ ...      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 状态管理

### Riverpod状态设计

```dart
/// 数据集管理状态
@freezed
class DatasetManagementState with _$DatasetManagementState {
  const factory DatasetManagementState.initial() = _DatasetManagementInitial;
  const factory DatasetManagementState.loading() = _DatasetManagementLoading;
  const factory DatasetManagementState.loaded({
    required List<Dataset> datasets,
    required List<DatasetImage> images,
    required DatasetFilters filters,
  }) = _DatasetManagementLoaded;
  const factory DatasetManagementState.error(String message) = _DatasetManagementError;
}

/// 数据集管理状态Provider
final datasetManagementProvider = StateNotifierProvider<DatasetManagementNotifier, DatasetManagementState>((ref) {
  return DatasetManagementNotifier(ref.read(datasetManagementRepositoryProvider));
});

/// 数据集管理状态管理器
class DatasetManagementNotifier extends StateNotifier<DatasetManagementState> {
  final DatasetManagementRepository _repository;

  DatasetManagementNotifier(this._repository) : super(const DatasetManagementState.initial());

  /// 加载数据集列表
  Future<void> loadDatasets({bool forceRefresh = false}) async {
    state = const DatasetManagementState.loading();
    try {
      final datasets = await _repository.getDatasetList();
      final images = await _repository.getAllDatasetImages();

      state = DatasetManagementState.loaded(
        datasets: datasets,
        images: images,
        filters: const DatasetFilters(),
      );
    } catch (e) {
      state = DatasetManagementState.error('加载数据集失败: ${e.toString()}');
    }
  }

  /// 应用筛选
  void applyFilters(DatasetFilters filters) {
    final currentState = state;
    if (currentState is _DatasetManagementLoaded) {
      state = currentState.copyWith(filters: filters);
    }
  }

  /// 创建新数据集
  Future<void> createDataset(Dataset dataset) async {
    try {
      await _repository.createDataset(dataset);
      await loadDatasets(); // 重新加载数据集
    } catch (e) {
      final currentState = state;
      state = DatasetManagementState.error('创建数据集失败: ${e.toString()}');
      state = currentState; // 恢复原状态
    }
  }

  /// 上传图像到数据集
  Future<void> uploadImagesToDataset(String datasetId, List<File> imageFiles) async {
    try {
      await _repository.uploadImagesToDataset(datasetId, imageFiles);
      await loadDatasets(); // 重新加载
    } catch (e) {
      final currentState = state;
      state = DatasetManagementState.error('上传图像失败: ${e.toString()}');
      state = currentState; // 恢复原状态
    }
  }

  /// 批量处理图像
  Future<void> batchProcessImages({
    required List<String> imageIds,
    required String algorithmId,
    required Map<String, dynamic> parameters,
  }) async {
    try {
      await _repository.batchProcessImages(
        imageIds: imageIds,
        algorithmId: algorithmId,
        parameters: parameters,
      );
      await loadDatasets(); // 重新加载
    } catch (e) {
      final currentState = state;
      state = DatasetManagementState.error('批量处理失败: ${e.toString()}');
      state = currentState; // 恢复原状态
    }
  }

  /// 删除数据集
  Future<void> deleteDataset(String datasetId) async {
    try {
      await _repository.deleteDataset(datasetId);
      await loadDatasets(); // 重新加载数据集
    } catch (e) {
      final currentState = state;
      state = DatasetManagementState.error('删除数据集失败: ${e.toString()}');
      state = currentState; // 恢复原状态
    }
  }
}

/// 仓储Provider
final datasetManagementRepositoryProvider = Provider<DatasetManagementRepository>((ref) {
  return DatasetManagementRepositoryImpl(
    ref.read(datasetManagementDatasourceProvider),
    ref.read(fileStorageProvider),
  );
});

/// 文件存储Provider
final fileStorageProvider = Provider<FileStorageService>((ref) {
  return FileStorageServiceImpl();
});

/// 单个数据集Provider
final datasetProvider = FutureProvider.family<Dataset, String>((ref, datasetId) async {
  return ref.read(datasetManagementRepositoryProvider).getDatasetById(datasetId);
});
```

---

## 📊 性能监控

### 关键性能指标

| 指标名称 | 目标值 | 监控方法 |
|---------|--------|---------|
| **数据集加载** | < 1s | 性能监控 |
| **瀑布流渲染** | < 500ms | UI性能监控 |
| **批量选择响应** | < 200ms | 交互监控 |
| **图片加载速度** | < 300ms/张 | 加载监控 |

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**维护团队**: Flutter开发团队