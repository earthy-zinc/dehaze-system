# 评估模块架构说明

## 数据流：UI → ViewModel → Repository → Data Source

### 1. UI 层 (Activity/Fragment)
- 负责展示数据和处理用户交互
- 通过 ViewModel 获取数据状态
- 使用 LiveData 观察数据变化

### 2. ViewModel 层
- 管理 UI 相关的数据
- 处理 UI 逻辑，如加载状态、错误处理等
- 通过 Repository 获取数据

### 3. Repository 层
- 处理数据逻辑，协调不同数据源
- 提供统一的数据访问接口
- 处理数据转换和错误处理

### 4. Data Source 层
- 实际的数据来源，如网络请求、本地数据库等
- 使用 dehaze-sdk-android 进行网络请求

## 示例数据流

```
EvaluationActivity (UI)
    ↓ (观察 LiveData)
EvaluationViewModel (状态管理)
    ↓ (调用方法)
EvaluationRepository (数据逻辑)
    ↓ (调用 SDK)
ModelAPI (网络请求)
AlgorithmAPI (网络请求)
```

## 组件说明

### Activity/Fragment
- `EvaluationActivity` - 评估主页面
- `ImageUploadFragment` - 图像上传页面
- `ResultFragment` - 评估结果页面

### ViewModel
- `EvaluationViewModel` - 管理评估相关的 UI 状态

### Repository
- `EvaluationRepository` - 处理评估数据相关的业务逻辑

### Adapter
- `MetricAdapter` - RecyclerView 的适配器，用于展示评估指标

## 使用的技术组件

1. **Navigation Component** - 页面导航
2. **ViewModel + LiveData** - 状态管理
3. **RecyclerView** - 评估指标列表展示
4. **ViewPager2** - 图像对比切换
5. **Glide** - 图片加载
6. **PhotoView** - 图片缩放
7. **dehaze-sdk-android** - 网络请求