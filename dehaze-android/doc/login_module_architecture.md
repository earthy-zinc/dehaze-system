# 登录模块架构说明

## 架构模式

本模块采用 MVVM（Model-View-ViewModel）架构模式，遵循 Android 官方推荐的架构指南。

### 组件说明

1. **View（视图层）**
   - [LoginFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginFragment.java#L16-L110)：负责展示UI界面和处理用户交互
   - 使用 ViewBinding 进行视图绑定，避免 findViewById 操作
   - 通过 DataBinding 实现与 ViewModel 的数据绑定

2. **ViewModel（视图模型层）**
   - [LoginViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginViewModel.java#L15-L197)：负责处理UI相关的业务逻辑
   - 使用 LiveData 管理UI状态
   - 与 Repository 层交互获取数据

3. **Repository（数据仓库层）**
   - 直接使用 dehaze-sdk-android 提供的 API
   - [AuthAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/AuthAPI.java#L1-L62)：处理认证相关请求（登录、获取验证码等）

4. **Model（数据模型层）**
   - 使用 dehaze-sdk-android 中定义的数据模型
   - [LoginRequest](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/model/auth/LoginRequest.java#L1-L13)：登录请求数据模型
   - [CaptchaResponse](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/model/auth/CaptchaResponse.java#L1-L12)：验证码响应数据模型

## 数据流说明

```
UI (LoginFragment) → ViewModel (LoginViewModel) → Repository (AuthAPI) → Model (SDK Models)
```

1. 用户在 [LoginFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginFragment.java#L16-L110) 中输入登录信息
2. [LoginFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginFragment.java#L16-L110) 通过 DataBinding 将数据同步到 [LoginViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginViewModel.java#L15-L197)
3. [LoginViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginViewModel.java#L15-L197) 处理登录逻辑，调用 [AuthAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/AuthAPI.java#L1-L62) 发起登录请求
4. [AuthAPI](file:///E:/DehazeSystem/dehaze-tool/dehaze-sdk-android/src/main/java/com/pei/dehaze/sdk/api/AuthAPI.java#L1-L62) 通过 Retrofit 发送网络请求
5. 请求结果通过回调返回给 [LoginViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginViewModel.java#L15-L197)
6. [LoginViewModel](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginViewModel.java#L15-L197) 更新 LiveData 状态
7. [LoginFragment](file:///E:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/ui/login/LoginFragment.java#L16-L110) 观察 LiveData 状态变化并更新UI

## 技术要点

1. **数据绑定**：使用 Android DataBinding 库实现 View 和 ViewModel 的双向绑定
2. **生命周期感知**：使用 LiveData 和 ViewModel 确保数据在配置变更时不会丢失
3. **异步处理**：网络请求在后台线程执行，结果通过回调返回主线程
4. **错误处理**：统一处理网络错误和业务错误，并通过 Toast 提示用户
5. **资源管理**：正确处理图片资源和内存泄漏问题

## UI 组件

1. **Material Design 组件**
   - TextInputLayout + TextInputEditText：输入框
   - MaterialButton：登录按钮
   - SwitchMaterial：主题切换开关
   - MaterialTextView：文本显示

2. **第三方库**
   - Glide：图片加载和显示
   - Retrofit + OkHttp：网络请求（封装在 SDK 中）

## 功能实现

1. **登录表单**
   - 用户名/密码输入验证
   - 验证码输入
   - 登录按钮状态管理

2. **验证码**
   - 获取并显示验证码图片
   - 点击刷新验证码

3. **登录流程**
   - 表单验证
   - 发起登录请求
   - 处理登录结果
   - 错误提示

4. **主题切换**
   - 深色/浅色主题切换
   - 使用 AppCompatDelegate 管理主题

5. **数据持久化**
   - 用户输入数据通过 ViewModel 管理
   - 验证码等临时数据使用 LiveData 管理