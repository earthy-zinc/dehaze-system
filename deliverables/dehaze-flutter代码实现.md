用户要求我根据设计文档完善 dehaze_flutter 项目并接入真实后端。我需要先阅读相关文档了解系统架构和模块设计，然后查看现有的 flutter 项目结构，最后给出整体代码实现方案。

让我先并行读取：
1. 系统架构文档目录
2. 模块设计文档目录
3. flutter 项目当前结构

让我先探索这些目录结构。
我先并行探索设计文档和现有 Flutter 项目的结构，了解全貌后再给出实现方案。
让我进一步探索模块设计文档、flutter 项目的详细结构和核心架构文档。
让我继续探索关键文档和现有代码结构。我需要读取：
1. 总体架构设计
2. API规范
3. 数据库设计
4. 前端额外说明
5. 现有 Flutter 项目的关键文件

同时探索现有的 flutter 项目结构。
现在并行读取关键架构文档和现有 Flutter 代码：
让我继续读取关键文件来理解现有 Flutter 项目的实现状态，以及关键模块文档。我需要：
1. 读取现有的 providers、interceptor、router
2. 读取关键模块设计文档
3. 读取数据库设计
4. 查看 pages 目录下的页面实现

让我并行读取这些文件。
继续并行探索现有 Flutter 代码和关键模块文档：
Let me continue exploring the Flutter project structure and read key module design documents. I need to understand:
1. The current state of image_input and dataset pages
2. The key module design documents
3. The database design

Let me read these in parallel.
继续读取关键页面实现和模块设计文档：
Let me now read the key module design documents to understand the API structure better. I need to look at:
1. 认证管理 module
2. 数据集管理 module
3. 算法管理 module
4. 文件管理 module

Let me also look at the existing image_input and dataset models/services to understand the current state.
继续读取关键模块设计文档和现有 models/services：
Now let me read the key module API documents to understand the API endpoints, and the dataset provider/service to understand current implementation.
现在并行读取关键模块的 API 接口文档和现有的 dataset provider/service：
Now I have a good understanding of the project. Let me also check the file management API and the 去雾处理 module, and read the README of dehaze_flutter to understand the project's documentation.
最后读取文件管理 API、去雾处理模块和 Flutter 项目的 README：
Now I have a comprehensive understanding of the project. Let me also check the去雾处理 requirements and the dehaze-sdk-js to understand the API structure better, since the docs mention that Flutter should use a similar approach.

Let me now synthesize all the information and create a comprehensive implementation plan.

Based on my analysis:

**Current State of dehaze_flutter:**
1. **Skeleton stage** - only basic UI structure exists
2. Has 3 pages: home (展示型), image_input, dataset
3. Uses Riverpod + GoRouter + Dio
4. API layer is minimal:
   - `api_config.dart`: baseUrl = `http://localhost:8080` (wrong, should be 8989)
   - `api_service.dart`: basic Dio setup
   - `auth_interceptor.dart`: empty (only has empty onError)
5. No authentication system (no login page, no token management)
6. No real API integration - all services fall back to mock data
7. Dataset and image_input services use mock data when connection fails
8. Response code check is wrong: uses `code == 0` but spec says `code == "00000"` (string)
9. Pagination params wrong: uses `page`/`page_size` but spec says `pageNum`/`pageSize`
10. No models for: User, Auth, Algorithm, File, Task, etc.

**Key Issues to Fix:**
1. API base URL should be `http://localhost:8989/api/v1` (per API spec)
2. Response format: `code` is string `"00000"` not int `0`
3. Pagination params: `pageNum`/`pageSize` not `page`/`page_size`
4. Response data structure: `{list: [], total: N}` not `{list, total, page, pageSize, totalPages}`
5. Auth: need login/logout/refresh/me flow with JWT token
6. Need complete interceptor: request (add token), response (handle code, 401 redirect), error handling

**What needs to be built:**

### 1. Core Infrastructure (核心基础设施)
- **API Layer**:
  - Fix `api_config.dart` - correct base URL, port 8989
  - Rewrite `auth_interceptor.dart` - full request/response/error interceptors
  - Add `response_interceptor.dart` - unified response parsing
  - Add `error_interceptor.dart` - 401 redirect, token refresh
  - Add `api_client.dart` - unified API response models

- **Models** (shared/common):
  - `api_result.dart` - unified response wrapper (code/msg/data/traceId/timestamp/errors)
  - `page_result.dart` - pagination response (list/total)
  - `user_model.dart` - user info
  - `auth_model.dart` - login request/response (accessToken, tokenType, expires)
  - `algorithm_model.dart` - algorithm info
  - `file_model.dart` - file metadata
  - `task_model.dart` - async task

- **Auth System**:
  - `auth_service.dart` - login/logout/refresh/getCaptcha/getMe
  - `auth_provider.dart` - auth state management
  - `token_storage.dart` - token persistence (SharedPreferences)
  - Login page
  - Route guards (redirect to login if not authenticated)

### 2. Feature Modules (功能模块)

**A. 认证管理 (Auth)**
- Login page (username/password, captcha)
- Auth state management
- Token refresh mechanism
- Route protection

**B. 图像输入 (Image Input)** - existing, needs real API
- Upload image to `/api/v1/files`
- Get samples from real backend (or keep as demo data)
- History from prediction logs

**C. 算法选择 (Algorithm Select)** - new
- GET `/api/v1/algorithm/options`
- Display algorithm tree/list
- Select algorithm for processing

**D. 去雾处理 (Dehaze Processing)** - new
- POST `/api/v1/prediction` with image + algorithm
- Polling task status GET `/api/v1/prediction/{taskId}`
- Show progress
- Display result

**E. 效果对比 (Effect Comparison)** - new
- Side-by-side comparison
- Overlay comparison
- Metrics display (PSNR/SSIM)
- POST `/api/v1/evaluation` for metrics

**F. 数据集管理 (Dataset Management)** - existing, needs real API
- Fix API calls to match spec
- GET `/api/v1/datasets` (tree)
- GET `/api/v1/dataset-items` (page)
- GET `/api/v1/item-files/{id}`

**G. 个人中心 (Profile)** - new
- User info display
- GET `/api/v1/users/me`
- Logout

### 3. Architecture Improvements
- Restructure to feature-based architecture
- Shared models in `lib/models/`
- Shared services in `lib/services/`
- Shared providers in `lib/providers/`

Let me now write up the comprehensive implementation plan. I'll also need to check the dehaze-sdk-js to understand the exact API models, since the docs mention Flutter should follow a similar pattern.

Actually, I have enough information now. Let me write the plan. Let me also check if there's a memory file to read first.
现在我已经全面了解了项目现状。让我检查一下是否有现有的记忆文件，然后给出完整的实现方案。
Now I have a comprehensive understanding. The memory shows that dehaze-uniapp was recently completed to ~98%. Now the user wants to do the same for dehaze_flutter. 

Key insights from memory:
1. Backend ports: Java 8989, Go 8999, Python 8014
2. Login: admin/123456
3. API response code is string `"00000"` not int `0`
4. Use `127.0.0.1` not `localhost` on Windows
5. The uniapp project already went through this exact process - I can reference its structure

Let me also look at the dehaze-uniapp's API structure to understand what was built, since it's the most recent reference implementation.

Now let me formulate the comprehensive implementation plan for dehaze_flutter. Let me summarize the current state and the plan.

Current state of dehaze_flutter:
- Skeleton UI with 3 pages (home, image_input, dataset)
- API layer minimal and incorrect (wrong port, wrong response code check, wrong pagination params)
- No auth system
- No real backend integration (all mock data)
- AuthInterceptor is empty
- Response code check `code == 0` should be `code == "00000"`
- Pagination params `page`/`page_size` should be `pageNum`/`pageSize`
- Missing pages: login, algorithm-select, processing, side-by-side, overlay, magnifier, filter, metrics, algorithm info, profile, task-history, file-manage

The plan should follow the same successful pattern as dehaze-uniapp but adapted for Flutter/Dart/Riverpod.

Let me write the plan now.
现在我已经全面了解了项目现状。基于对设计文档、现有 Flutter 代码、以及刚完成的 dehaze-uniapp 项目（作为参考实现）的深入分析，下面是 **dehaze_flutter 完善与接入真实后端的整体代码实现方案**。

---

# Dehaze Flutter 完善与后端接入 — 整体实现方案

## 一、现状分析

### 1.1 已有基础

| 模块 | 状态 | 说明 |
|------|------|------|
| 项目脚手架 | ✅ | Flutter 3.35 + Dart 3.9，6 平台目录完整 |
| 主题系统 | ✅ | `app_theme.dart` 完整的明暗主题 |
| 响应式工具 | ✅ | `responsive_utils.dart` 完整 |
| 主布局 | ✅ | `main_layout.dart` 侧边栏+底部导航+抽屉 |
| 路由 | ⚠️ 部分 | GoRouter 已配置，但无路由守卫、无登录页 |
| 首页 | ✅ | 展示型 Hero+功能区块 |
| 图像输入 | ⚠️ UI完整 | 4 种输入方式 UI 已实现，但使用 Mock 数据 |
| 数据集管理 | ⚠️ UI完整 | 列表+详情 UI 已实现，但使用 Mock 数据 |
| API 层 | ❌ | baseUrl 端口错误(8080)、响应码判断错误(`==0`)、分页参数错误、AuthInterceptor 为空 |
| 认证系统 | ❌ | 无登录页、无 Token 管理、无路由守卫 |
| 算法选择 | ❌ | 未实现 |
| 去雾处理 | ❌ | 未实现 |
| 效果对比(6页) | ❌ | 未实现 |
| 用户中心 | ❌ | 未实现 |
| 处理历史 | ❌ | 未实现 |

### 1.2 核心问题清单

| # | 问题 | 位置 | 严重度 |
|---|------|------|--------|
| 1 | baseUrl 端口 `8080`，应为 `8989` | `api_config.dart:3` | 🔴 |
| 2 | 响应码判断 `code == 0`，应为 `code == "00000"`（字符串） | `dataset_service.dart:25`、`image_input_service.dart:183` | 🔴 |
| 3 | 分页参数 `page`/`page_size`，应为 `pageNum`/`pageSize` | `dataset_service.dart:18-19` | 🔴 |
| 4 | 分页响应结构错误，后端只返回 `{list, total}`，无 `page/pageSize/totalPages` | `dataset_model.dart:82-109` | 🔴 |
| 5 | AuthInterceptor 只有空 `onError`，无请求拦截(注入Token)、无响应拦截(code处理)、无401重定向 | `auth_interceptor.dart` | 🔴 |
| 6 | 图片类型枚举 `foggy/clear/annotated`，应为 `hazy/clear/dehazed` | `dataset_model.dart:5-12` | 🟡 |
| 7 | 无 Token 持久化、无登录态管理 | 全局 | 🔴 |
| 8 | Windows 环境应用 `127.0.0.1` 而非 `localhost` | `api_config.dart:3` | 🟡 |

---

## 二、目标架构

### 2.1 目录结构（重构后）

```
lib/
├── main.dart                          # 入口：初始化 + ProviderScope
├── app.dart                           # MaterialApp.router 配置
│
├── core/                              # 核心基础设施（新增）
│   ├── network/                       # 网络层
│   │   ├── api_config.dart            # ★ 修正：baseUrl=127.0.0.1:8989/api/v1
│   │   ├── api_client.dart            # ★ 新增：Dio 单例 + 拦截器注册
│   │   ├── api_result.dart            # ★ 新增：统一响应包装 {code,msg,data,traceId}
│   │   ├── page_result.dart           # ★ 新增：分页响应 {list,total}
│   │   └── interceptors/
│   │       ├── auth_interceptor.dart  # ★ 重写：请求注入Token + 401处理
│   │       ├── response_interceptor.dart  # ★ 新增：统一 code 判断 + 业务错误
│   │       ├── error_interceptor.dart # ★ 新增：网络错误/超时处理
│   │       └── retry_interceptor.dart # ★ 新增：Token 刷新并发队列
│   ├── storage/
│   │   └── token_storage.dart         # ★ 新增：Token 读写(SharedPreferences)
│   ├── constants/
│   │   ├── api_constants.dart         # ★ 新增：API路径常量
│   │   └── storage_constants.dart     # ★ 新增：存储Key常量
│   └── utils/
│       ├── responsive_utils.dart      # 已有
│       └── date_utils.dart            # ★ 新增：日期格式化
│
├── models/                            # 共享数据模型（新增）
│   ├── user_model.dart                # 用户信息
│   ├── auth_model.dart                # 登录请求/响应
│   ├── algorithm_model.dart           # 算法信息
│   ├── prediction_model.dart          # 预测任务/结果
│   ├── evaluation_model.dart          # 评估指标
│   ├── file_model.dart                # 文件元数据
│   ├── dataset_model.dart             # ★ 迁移：从pages/dataset/models/移到共享
│   └── task_model.dart                # 异步任务
│
├── services/                          # API 服务层（新增）
│   ├── auth_service.dart              # /auth/login,logout,captcha,refresh,me
│   ├── file_service.dart              # /files 上传/下载/查询
│   ├── algorithm_service.dart         # /algorithm 列表/详情/options
│   ├── prediction_service.dart        # /prediction 执行/状态/日志
│   ├── evaluation_service.dart        # /evaluation 执行/状态/日志
│   ├── dataset_service.dart           # ★ 迁移+修正：/datasets,/dataset-items
│   └── task_service.dart              # /tasks 统一任务接口
│
├── providers/                         # 全局 Riverpod Providers
│   ├── providers.dart                 # ★ 修正：基础Provider
│   ├── auth_provider.dart             # ★ 新增：认证状态管理
│   ├── user_provider.dart             # ★ 新增：用户信息
│   └── home_provider.dart             # 已有
│
├── router/                            # 路由
│   ├── config.dart                    # ★ 修正：添加登录页+路由守卫
│   └── route_guard.dart               # ★ 新增：登录态重定向
│
├── layout/
│   ├── main_layout.dart               # ★ 修正：添加用户信息显示
│   └── menu_config.dart               # ★ 修正：添加新页面菜单项
│
├── theme/
│   └── app_theme.dart                 # 已有
│
├── constants/
│   └── app_constants.dart             # 已有
│
├── widgets/                           # 共享组件
│   ├── loading_widget.dart            # ★ 新增
│   ├── error_widget.dart              # ★ 新增
│   ├── empty_widget.dart              # ★ 新增
│   ├── cached_image.dart              # ★ 新增
│   └── confirm_dialog.dart            # ★ 新增
│
└── pages/                             # 功能页面
    ├── home/                          # 已有（展示型）
    ├── login/                         # ★ 新增：登录页
    ├── image_input/                   # ★ 修正：接入真实上传API
    │   ├── index.dart
    │   ├── models/
    │   ├── providers/
    │   ├── services/
    │   └── widgets/
    ├── algorithm_select/              # ★ 新增：算法选择页
    ├── processing/                    # ★ 新增：去雾处理页
    ├── comparison/                    # ★ 新增：效果对比(6个子页面)
    │   ├── side_by_side.dart
    │   ├── overlay.dart
    │   ├── magnifier.dart
    │   ├── filter.dart
    │   ├── metrics.dart
    │   └── algorithm_info.dart
    ├── dataset/                       # ★ 修正：接入真实API
    │   ├── index.dart
    │   ├── models/
    │   ├── providers/
    │   ├── services/
    │   └── widgets/
    ├── profile/                       # ★ 新增：用户中心
    ├── task_history/                  # ★ 新增：处理历史
    └── file_manage/                   # ★ 新增：文件管理
```

### 2.2 分层架构

```
┌─────────────────────────────────────────────────────┐
│                    UI 层 (Pages)                     │
│  login | image_input | algorithm_select | processing│
│  comparison | dataset | profile | task_history       │
├─────────────────────────────────────────────────────┤
│               状态管理层 (Riverpod)                  │
│  auth_provider | user_provider | processing_provider│
│  dataset_provider | algorithm_provider              │
├─────────────────────────────────────────────────────┤
│                服务层 (Services)                     │
│  auth_service | file_service | algorithm_service    │
│  prediction_service | dataset_service               │
├─────────────────────────────────────────────────────┤
│              网络层 (Core/Network)                   │
│  api_client(Dio) + interceptors + api_result        │
├─────────────────────────────────────────────────────┤
│              模型层 (Models)                         │
│  user | algorithm | prediction | dataset | file     │
└─────────────────────────────────────────────────────┘
```

---

## 三、分阶段实施计划

### Phase 0: 核心基础设施修复（P0，必须最先完成）

**目标**：修复网络层，建立正确的 API 通信基础

| 任务 | 文件 | 说明 |
|------|------|------|
| T01 | `core/network/api_config.dart` | 修正 baseUrl 为 `http://127.0.0.1:8989/api/v1` |
| T02 | `core/network/api_result.dart` | 统一响应模型：`code`(String) + `msg` + `data` + `traceId` + `errors` |
| T03 | `core/network/page_result.dart` | 分页模型：`{list: List<T>, total: int}` |
| T04 | `core/network/api_client.dart` | Dio 单例，注册所有拦截器 |
| T05 | `core/storage/token_storage.dart` | Token 持久化（accessToken/refreshToken） |
| T06 | `core/network/interceptors/auth_interceptor.dart` | **重写**：请求注入 `Authorization: Bearer <token>` |
| T07 | `core/network/interceptors/response_interceptor.dart` | 统一判断 `code == "00000"`，业务错误抛出 `ApiException` |
| T08 | `core/network/interceptors/error_interceptor.dart` | 401 → 跳转登录页；超时/网络错误友好提示 |
| T09 | `core/network/interceptors/retry_interceptor.dart` | Token 过期自动刷新 + 并发请求队列 |
| T10 | `core/constants/api_constants.dart` | 所有 API 路径常量（`/auth/login` 等） |

### Phase 1: 认证系统（P0）

**目标**：完整 JWT 认证流程

| 任务 | 文件 | 说明 |
|------|------|------|
| T11 | `models/auth_model.dart` | `LoginRequest`、`LoginResponse`(accessToken,tokenType,expires)、`CaptchaResponse` |
| T12 | `models/user_model.dart` | 用户信息模型（id,username,nickname,roles,permissions） |
| T13 | `services/auth_service.dart` | `login()`、`logout()`、`getCaptcha()`、`refreshToken()`、`getCurrentUser()` |
| T14 | `providers/auth_provider.dart` | `AuthNotifier`：管理 token + 用户信息 + login/logout |
| T15 | `pages/login/index.dart` | 登录页：用户名+密码+验证码，表单校验，错误提示 |
| T16 | `router/route_guard.dart` | GoRouter refreshListenable 监听登录态，未登录重定向到 `/login` |
| T17 | `router/config.dart` | 注册登录页路由，白名单（login/home），其他页面需登录 |
| T18 | `layout/main_layout.dart` | 侧边栏底部显示用户信息+登出按钮 |

### Phase 2: 核心去雾流程（P1）

**目标**：打通「图像输入 → 算法选择 → 去雾处理 → 结果展示」主链路

| 任务 | 文件 | 说明 |
|------|------|------|
| T19 | `services/file_service.dart` | `uploadFile(MultipartFile)` → 返回 fileId+url；`checkMd5()` 秒传 |
| T20 | `models/algorithm_model.dart` | 算法树形结构（id,name,type,status,config） |
| T21 | `services/algorithm_service.dart` | `getAlgorithmOptions()`、`getAlgorithmDetail(id)` |
| T22 | `pages/algorithm_select/index.dart` | 算法列表页：卡片展示、选中高亮、底部确认按钮 |
| T23 | `models/prediction_model.dart` | `PredictionRequest`、`PredictionResponse`(taskId,status,resultUrl,duration) |
| T24 | `services/prediction_service.dart` | `predict()`、`getPredictionStatus(taskId)`、`getPredictionLogs()` |
| T25 | `providers/processing_provider.dart` | 处理流程状态机：image→algorithm→params→processing→result |
| T26 | `pages/processing/index.dart` | 参数调节滑块 + 调用预测API + 进度展示 + 结果对比入口 |
| T27 | `pages/image_input/` 修正 | 上传图片调用 `/files` 获取 fileId；清除 Mock 降级逻辑 |

### Phase 3: 效果对比模块（P1）

**目标**：6 种对比模式

| 任务 | 文件 | 说明 |
|------|------|------|
| T28 | `pages/comparison/side_by_side.dart` | 并排对比：触控滑动分割线 |
| T29 | `pages/comparison/overlay.dart` | 重叠对比：透明度滑块 + 预设 |
| T30 | `pages/comparison/magnifier.dart` | 放大镜：触控移动 + 镜片大小可调 |
| T31 | `pages/comparison/filter.dart` | 滤镜调节：亮度/对比度/饱和度/色温 + 预设 |
| T32 | `models/evaluation_model.dart` | 评估指标（PSNR,SSIM,MSE,FSIM,LPIPS） |
| T33 | `services/evaluation_service.dart` | `evaluate()`、`getEvaluationStatus()`、`getEvaluationLogs()` |
| T34 | `pages/comparison/metrics.dart` | 指标评估：调用评估API + 结果展示 |
| T35 | `pages/comparison/algorithm_info.dart` | 算法详情：从 `/algorithm/{id}` 加载 |
| T36 | `router/config.dart` | 注册 6 个对比页面路由 |

### Phase 4: 数据集管理接入真实API（P1）

**目标**：替换 Mock 数据，对接真实后端

| 任务 | 文件 | 说明 |
|------|------|------|
| T37 | `models/dataset_model.dart` | **修正**：图片类型改 `hazy/clear/dehazed`；分页响应只保留 `{list,total}` |
| T38 | `services/dataset_service.dart` | **重写**：参数改 `pageNum/pageSize`；响应码改 `"00000"`；删除全部 Mock |
| T39 | `pages/dataset/providers/dataset_provider.dart` | 适配新分页结构 |
| T40 | `pages/dataset/providers/image_provider.dart` | 适配新分页结构 |
| T41 | `pages/dataset/widgets/` | 适配新图片类型枚举 |

### Phase 5: 基础管理模块（P2）

| 任务 | 文件 | 说明 |
|------|------|------|
| T42 | `pages/profile/index.dart` | 用户中心：头像、昵称、角色标签、退出登录 |
| T43 | `pages/task_history/index.dart` | 处理历史：从 `/prediction/logs` 加载列表 |
| T44 | `pages/file_manage/index.dart` | 文件管理：从 `/files/page` 加载 |
| T45 | `layout/menu_config.dart` | 添加"我的"菜单组（用户中心+处理历史） |

### Phase 6: 收尾

| 任务 | 说明 |
|------|------|
| T46 | 全局搜索清除所有 Mock 数据降级逻辑 |
| T47 | `pubspec.yaml` 补充依赖（如 `web_socket_channel` 用于实时进度） |
| T48 | `README.md` 更新项目结构和运行说明 |
| T49 | `flutter analyze` 通过，无警告 |

---

## 四、关键技术决策

### 4.1 API 响应处理

```dart
// core/network/api_result.dart
class ApiResult<T> {
  final String code;        // "00000" = 成功
  final String msg;
  final T? data;
  final String? traceId;
  final List<ApiError>? errors;
  
  bool get isSuccess => code == '00000';
}

// 响应拦截器自动解包：业务层直接拿到 data，无需手动判断 code
```

### 4.2 Token 刷新策略

```
请求A → 401 → 暂停所有后续请求 → 调用 /auth/refresh
      → 成功：重发暂存请求
      → 失败：清除Token → 跳转登录页
```

### 4.3 图片类型枚举对齐

```dart
// 修正前（错误）          // 修正后（与后端一致）
foggy    →  hazy      // 有雾图像
clear    →  clear     // 清晰图像
annotated → dehazed   // 去雾结果
```

### 4.4 路由守卫

```dart
// GoRouter refreshListenable 监听 authProvider 状态
// 白名单：/login, /home（首页可未登录访问）
// 其他页面：未登录 → 重定向 /login
```

### 4.5 文件上传

```dart
// 使用 dio 的 MultipartFile
final formData = FormData.fromMap({
  'file': await MultipartFile.fromFile(imagePath, filename: filename),
});
final response = await dio.post('/files', data: formData);
// 返回 {fileId, url, name, size, md5}
```

---

## 五、实施优先级与依赖关系

```
Phase 0 (基础设施) ──→ Phase 1 (认证) ──→ Phase 2 (核心流程)
                                              │
                                              ├──→ Phase 3 (效果对比)
                                              ├──→ Phase 4 (数据集修正)
                                              └──→ Phase 5 (基础管理)
                                                       │
                                                       └──→ Phase 6 (收尾)
```

**建议执行顺序**：P0 → P1 → P2 → P4 → P3 → P5 → P6

> 先修数据集(P4)再做效果对比(P3)，因为数据集是现有代码只需修正，成本低、见效快。

---

## 六、预期最终状态

| 维度 | 当前 | 目标 |
|------|------|------|
| 注册页面数 | 3 | **15** |
| API 服务模块 | 0(全Mock) | **7** |
| 共享数据模型 | 0 | **8** |
| 认证流程 | 无 | 完整JWT |
| 路由守卫 | 无 | 完整 |
| Mock 数据 | 100% | **0%**（全部真实API） |
| 完整度 | ~15% | **~98%** |
