# 仪表盘模块架构说明

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
DashboardActivity (UI)
    ↓ (观察 LiveData)
DashboardViewModel (状态管理)
    ↓ (调用方法)
DashboardRepository (数据逻辑)
    ↓ (调用 SDK)
UserAPI (网络请求)
```

## 组件说明

### Activity/Fragment
- `DashboardActivity` - 仪表盘主页面
- `ChartFragment` - 图表展示页面

### ViewModel
- `DashboardViewModel` - 管理仪表盘相关的 UI 状态

### Repository
- `DashboardRepository` - 处理仪表盘数据相关的业务逻辑

### Adapter
- `StatAdapter` - RecyclerView 的适配器，用于展示统计数据卡片

## 使用的技术组件

1. **Navigation Component** - 页面导航
2. **ViewModel + LiveData** - 状态管理
3. **RecyclerView** - 统计数据列表展示
4. **MPAndroidChart** - 图表展示
5. **Glide** - 图片加载
6. **dehaze-sdk-android** - 网络请求