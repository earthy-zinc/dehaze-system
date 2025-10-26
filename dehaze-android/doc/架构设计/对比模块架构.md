# 对比模块架构说明

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
CompareActivity (UI)
    ↓ (观察 LiveData)
CompareViewModel (状态管理)
    ↓ (调用方法)
CompareRepository (数据逻辑)
    ↓ (调用 SDK)
ModelAPI (网络请求)
```

## 组件说明

### Activity/Fragment
- `CompareActivity` - 图像对比主页面
- `ParallelFragment` - 并排对比页面
- `OverlapFragment` - 重叠对比页面

### ViewModel
- `CompareViewModel` - 管理图像对比相关的 UI 状态

### Repository
- `CompareRepository` - 处理图像对比数据相关的业务逻辑

### 自定义View
- `MagnifierView` - 放大镜视图
- `OverlapImageView` - 重叠图像视图

## 使用的技术组件

1. **ViewPager2 + Fragment** - 页面切换
2. **ViewModel + LiveData** - 状态管理
3. **Glide** - 图片加载
4. **PhotoView** - 图片缩放
5. **Custom Views** - 自定义图像处理视图
6. **dehaze-sdk-android** - 网络请求