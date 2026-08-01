# Android 原生 (dehaze-android)

将 dehaze-front-vue（桌面 Web）的核心业务功能等价迁移为原生 Android 应用，使用 Java 语言开发。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 架构图

```mermaid
flowchart TB
    subgraph View["UI 层"]
        Activity["Activity"]
        Fragment["Fragment"]
        View["自定义 View"]
        Adapter["RecyclerView Adapter"]
    end

    subgraph Binding["数据绑定"]
        DataBinding["DataBinding"]
        ViewBinding["ViewBinding"]
    end

    subgraph ViewModel["ViewModel 层"]
        VM["ViewModel + LiveData"]
        Navigation["Navigation Component"]
    end

    subgraph Repository["Repository 层"]
        Repo["数据仓库"]
    end

    subgraph DataSource["数据源层"]
        API["网络 API (Retrofit2 + OkHttp3)"]
        SDK["SDK 层 (Token注入/拦截器)"]
        CameraX["CameraX"]
    end

    subgraph Backend["后端"]
        REST["RESTful API"]
    end

    View --> Binding --> VM
    VM --> Repo
    Repo --> API
    API --> SDK
    SDK --> REST
    Repo --> CameraX
    Navigation --> Activity
    Navigation --> Fragment
```

## 2. 架构设计

应用采用标准的 MVVM 架构模式：

- **View 层**：Activity、Fragment、自定义 View，通过 DataBinding 绑定 ViewModel
- **ViewModel 层**：LiveData 驱动 UI 状态管理，Navigation Component 管理页面导航
- **Repository 层**：数据仓库层，统一管理网络和本地数据源
- **Data Source 层**：网络 API（Retrofit2 + OkHttp3）和 CameraX 相机能力

## 3. 模块结构

```
com.pei.dehaze
├── ui                  # UI层
│   ├── login           # 登录模块
│   ├── register        # 注册模块
│   ├── profile         # 个人中心模块（含未登录态入口）
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

## 4. 核心功能

1. 用户认证 - 登录、注册、忘记密码
2. 数据集管理 - 查看和管理图像数据集
3. 算法管理 - 浏览和搜索去雾算法、智能推荐
4. 去雾处理 - 单张/批量去雾、参数调节、处理历史
5. 图像对比 - 并排和叠加方式对比处理前后图像、6种对比模式
6. 图像评估 - 上传图像并进行去雾处理，查看评估指标
7. 图像展示 - 实时演示不同算法的去雾效果
8. 收藏管理 - 跨模块统一收藏（算法/处理结果/数据集）、收藏聚合页
9. 推荐管理 - 推荐算法展示、推荐理由、一键使用
10. 系统管理 - 管理用户、部门和角色信息

## 5. 认证架构

采用 Session 认证：

- SessionId 存储：TokenManager 通过 TokenStorage 实现持久化，应用启动时自动恢复
- 请求鉴权：SDK 拦截器自动为非公开端点注入 X-Session-Id 请求头
- 7天免登录：登录页"记住我"复选框，勾选时 LoginRequest.rememberMe=true
- Session 失效处理：ApiCallback 在收到 401 或 A0230 业务码时触发 TokenManager.triggerSessionInvalid()，全局监听器通知当前 Activity 弹出"登录已失效"对话框并跳转登录页
- 未登录态展示：个人中心页面在 TokenManager.hasToken()=false 时显示"未登录"入口卡片，点击跳转登录/注册页

## 6. 主要组件

- Navigation Component - 应用内页面导航
- ViewModel + LiveData - UI 状态管理
- RecyclerView - 列表展示
- ViewPager2 - 页面滑动切换
- DataBinding - 数据绑定
- CameraX - 拍照功能

## 7. 权限说明

- INTERNET - 访问网络接口
- CAMERA - 拍照功能
- READ_EXTERNAL_STORAGE/WRITE_EXTERNAL_STORAGE - 读取和保存图像文件

## 8. 兼容性

- 最低支持 Android 6.0 (API Level 23)
- 支持 Android 14 (API Level 34)
- 屏幕适配：支持各种屏幕尺寸和分辨率

## 9. 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 架构模式 | MVVM | Android Jetpack 推荐，清晰分层 |
| 网络层 | Retrofit2 + OkHttp3 | 成熟的 Android HTTP 客户端 |
| 认证方案 | Session 认证 | 与后端三端统一 |
| 导航 | Navigation Component | Jetpack 官方导航方案 |
| 相机 | CameraX | Jetpack 相机库，向后兼容 |
| UI 设计 | Material Design 3 | 遵循 Google 人机交互指南 |
| 收藏状态同步 | ViewModel + LiveData + Room 本地缓存 | Android 端通过 ViewModel 管理收藏状态，Room 数据库缓存收藏列表支持离线浏览；收藏按钮使用 ImageButton + 视图动画反馈 |
| 推荐图片上传 | ContentResolver + Glide 压缩 | Android 端通过 ContentResolver 访问相册，Glide 加载时自动按 ImageView 尺寸压缩，上传前通过 BitmapFactory 进一步压缩至 5MB 内 |

## 10. 模块间交互

- 通过 RESTful API 调用 Java/Go/Python 后端
- 项目内置 SDK 封装网络层（Retrofit2 + OkHttp3），统一 Token 注入与错误处理
