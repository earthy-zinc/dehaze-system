# 展示模块架构说明

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
PresentationActivity (UI)
    ↓ (观察 LiveData)
PresentationViewModel (状态管理)
    ↓ (调用方法)
PresentationRepository (数据逻辑)
    ↓ (调用 SDK)
AlgorithmAPI (网络请求)
ModelAPI (网络请求)
FileAPI (网络请求)
```

## 组件说明

### Activity/Fragment
- `PresentationActivity` - 展示主页面
- `CameraFragment` - 拍照页面
- `ImageDisplayFragment` - 图像展示页面
- `ImageComparisonFragment` - 图像对比页面

### ViewModel
- `PresentationViewModel` - 管理展示相关的 UI 状态

### Repository
- `PresentationRepository` - 处理展示数据相关的业务逻辑

### Adapter
- `ExampleImageAdapter` - RecyclerView 的适配器，用于展示样例图片
- `AlgorithmAdapter` - RecyclerView 的适配器，用于展示算法列表

## 使用的技术组件

1. **Navigation Component** - 页面导航
2. **ViewModel + LiveData** - 状态管理
3. **RecyclerView** - 样例图片和算法列表展示
4. **ViewPager2** - 图像展示切换
5. **CameraX** - 拍照功能
6. **Glide** - 图片加载
7. **PhotoView** - 图片缩放
8. **dehaze-sdk-android** - 网络请求