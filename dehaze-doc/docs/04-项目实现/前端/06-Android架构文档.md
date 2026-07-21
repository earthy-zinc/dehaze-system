# Android 原生 (dehaze-android)

将 dehaze-front-vue（桌面 Web）的核心业务功能等价迁移为原生 Android 应用，使用 Java 语言开发。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 功能特性

1. **用户认证** - 登录、注册、忘记密码
2. **数据集管理** - 查看和管理图像数据集
3. **算法管理** - 浏览和搜索去雾算法
4. **图像对比** - 并排和叠加方式对比处理前后图像
5. **图像评估** - 上传图像并进行去雾处理，查看评估指标
6. **图像展示** - 实时演示不同算法的去雾效果
7. **系统管理** - 管理用户、部门和角色信息

## 2. 架构设计

应用采用标准的 MVVM 架构模式，分为以下层级：

```
┌─────────────┐
│    View     │ ← UI层 (Activity, Fragment, View)
└─────────────┘
       ↓
┌─────────────┐
│  ViewModel  │ ← 数据处理和UI逻辑层
└─────────────┘
       ↓
┌─────────────┐
│ Repository  │ ← 数据仓库层
└─────────────┘
       ↓
┌─────────────┐
│ Data Source │ ← 数据源 (网络API, 本地数据库)
└─────────────┘
```

## 3. 模块结构

```
com.pei.dehaze
├── ui                  # UI层
│   ├── login           # 登录模块
│   ├── dashboard       # 仪表盘模块
│   ├── dataset         # 数据集模块
│   ├── algorithm       # 算法模块
│   ├── compare         # 图像对比模块
│   ├── evaluation      # 图像评估模块
│   ├── presentation    # 图像展示模块
│   └── system          # 系统管理模块
├── repository          # 数据仓库层
├── model               # 数据模型
├── network             # 网络层
├── utils               # 工具类
├── common              # 公共组件
└── sdk                 # SDK封装
```

## 4. 主要组件

- Navigation Component - 应用内页面导航
- ViewModel + LiveData - UI 状态管理
- RecyclerView - 列表展示
- ViewPager2 - 页面滑动切换
- DataBinding - 数据绑定
- CameraX - 拍照功能

## 5. 权限说明

应用需要以下权限：

- INTERNET - 访问网络接口
- CAMERA - 拍照功能
- READ_EXTERNAL_STORAGE/WRITE_EXTERNAL_STORAGE - 读取和保存图像文件

## 6. 兼容性

- 最低支持 Android 6.0 (API Level 23)
- 支持 Android 14 (API Level 34)
- 屏幕适配：支持各种屏幕尺寸和分辨率

## 7. SDK 说明

项目内置一个基于 Retrofit2、OkHttp3、Lombok 和 Timber 的 Android SDK，位于 [dehaze-android/sdk](../../dehaze-android/sdk) 目录。SDK 的功能特点、使用方式、初始化代码、依赖版本等详见 [sdk/README.md](../../dehaze-android/sdk/README.md)。

## 8. 项目成果

### 8.1 功能完整性

完整迁移了 Web 端的所有核心功能：

- 用户认证、数据集管理、图像上传、去雾处理
- 结果对比、指标评估等核心业务
- 完善的系统管理功能

### 8.2 体验合规性

- 遵循 Android Material Design 3 与 Google 人机交互指南
- 重构了桌面端交互（如拖拽 → 滑动切换，画布 → 图片预览）
- 充分利用了 Android 原生能力

### 8.3 架构现代化

- 采用 Android Jetpack 架构组件实现清晰分层
- 使用 Navigation、Room、DataStore、ViewModel + LiveData 等现代化组件
- 实现了良好的代码组织和模块划分

## 9. 后续建议

1. **持续集成/持续部署(CI/CD)**
   - 设置自动化构建和测试流程
   - 集成代码质量检查工具

2. **性能优化**
   - 进一步优化图片加载和处理性能
   - 减少内存占用和电池消耗

3. **功能扩展**
   - 增加更多去雾算法支持
   - 实现离线处理能力

4. **用户体验优化**
   - 增加夜间模式支持
   - 优化动画效果和过渡体验
   - 提供更多个性化设置选项
