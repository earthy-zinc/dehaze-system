# 数据集模块架构说明

## 架构模式

本模块采用 MVVM（Model-View-ViewModel）架构模式，遵循 Android 官方推荐的架构指南。

### 组件说明

1. **View（视图层）**
   - [DatasetDetailFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetDetailFragment.java#L15-L169)：负责展示数据集详情UI界面和处理用户交互
   - 使用 ViewBinding 进行视图绑定
   - 通过 DataBinding 实现与 ViewModel 的数据绑定

2. **ViewModel（视图模型层）**
   - [DatasetViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetViewModel.java#L16-L303)：负责处理UI相关的业务逻辑
   - 使用 LiveData 管理UI状态
   - 与 Repository 层交互获取数据

3. **Repository（数据仓库层）**
   - 直接使用 dehaze-sdk-android 提供的 API
   - [DatasetAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/DatasetAPI.java#L15-L183)：处理数据集相关请求

4. **Model（数据模型层）**
   - 使用 dehaze-sdk-android 中定义的数据模型
   - [Dataset](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/model/dataset/Dataset.java#L1-L70)：数据集数据模型
   - [ImageItem](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/model/dataset/ImageItem.java#L1-L12)：图片项数据模型
   - [ImageUrl](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/model/dataset/ImageUrl.java#L1-L30)：图片URL数据模型
   - 自定义模型：[ViewCard](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/model/ViewCard.java#L1-L54)、[ImageType](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/model/ImageType.java#L1-L44)

## 数据流说明

```
UI (DatasetDetailFragment) → ViewModel (DatasetViewModel) → Repository (DatasetAPI) → Model (SDK Models)
```

1. [DatasetDetailFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetDetailFragment.java#L15-L169) 通过 DataBinding 与 [DatasetViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetViewModel.java#L16-L303) 进行数据绑定
2. [DatasetViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetViewModel.java#L16-L303) 处理业务逻辑，调用 [DatasetAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/DatasetAPI.java#L15-L183) 获取数据
3. [DatasetAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/DatasetAPI.java#L15-L183) 通过 Retrofit 发送网络请求
4. 请求结果通过回调返回给 [DatasetViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetViewModel.java#L16-L303)
5. [DatasetViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetViewModel.java#L16-L303) 更新 LiveData 状态
6. [DatasetDetailFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/dataset/DatasetDetailFragment.java#L15-L169) 观察 LiveData 状态变化并更新UI

## 技术要点

1. **数据绑定**：使用 Android DataBinding 库实现 View 和 ViewModel 的数据绑定
2. **生命周期感知**：使用 LiveData 和 ViewModel 确保数据在配置变更时不会丢失
3. **异步处理**：网络请求在后台线程执行，结果通过回调返回主线程
4. **错误处理**：统一处理网络错误和业务错误，并通过 Toast 提示用户
5. **图片加载**：使用 Glide 加载图片，支持占位图和错误图
6. **列表展示**：使用 RecyclerView + StaggeredGridLayoutManager 实现瀑布流布局
7. **下拉刷新**：使用 SwipeRefreshLayout 实现下拉刷新功能

## UI 组件

1. **Material Design 组件**
   - MaterialCardView：卡片布局
   - TextInputLayout + TextInputEditText：输入框
   - MaterialButton：按钮
   - SwipeRefreshLayout：下拉刷新

2. **第三方库**
   - Glide：图片加载和显示
   - RecyclerView：列表展示
   - StaggeredGridLayoutManager：瀑布流布局

## 功能实现

1. **数据集详情展示**
   - 数据集名称、类型、描述展示
   - 使用 DataBinding 实现数据绑定

2. **图片瀑布流展示**
   - 使用 RecyclerView + StaggeredGridLayoutManager
   - 自定义 Adapter 实现图片展示
   - Glide 加载图片

3. **图片类型切换**
   - 动态创建按钮
   - 点击切换图片类型
   - 更新图片列表

4. **图片搜索**
   - 搜索框输入关键词
   - 点击搜索按钮执行搜索
   - 重置按钮清空搜索条件

5. **下拉刷新**
   - 使用 SwipeRefreshLayout
   - 刷新时重新加载数据

6. **分页加载**
   - 支持加载更多数据
   - 通过 ViewModel 管理分页状态

## 架构优势

1. **关注点分离**：各层职责明确，便于维护和扩展
2. **可测试性**：ViewModel 便于单元测试
3. **生命周期管理**：避免内存泄漏
4. **数据一致性**：通过 LiveData 确保数据一致性
5. **响应式编程**：使用 LiveData 实现响应式UI更新