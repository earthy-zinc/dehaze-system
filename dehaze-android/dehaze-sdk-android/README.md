# Dehaze Android SDK

Android SDK for Dehaze System API based on Retrofit2, OkHttp3, Lombok and Timber.

## 功能特点

- 基于 Retrofit2 和 OkHttp3 实现网络请求
- 使用 Lombok 简化 Java 代码
- 集成 Timber 日志框架
- 提供完整的 API 接口封装
- 支持异步回调处理
- 支持文件上传和下载
- 自动处理 Token 认证
- 模块化 API 设计

## 本地依赖安装方式

由于本SDK仅用于当前项目，不需要上传到公共仓库，因此提供了两种本地依赖方式：

### 方式一：作为模块导入（推荐）

1. 将 `dehaze-sdk-android` 目录复制到你的 Android 项目根目录下
2. 在项目根目录的 `settings.gradle` 文件中添加：

```gradle
include ':dehaze-sdk-android'
```

3. 在需要使用 SDK 的模块（如 app 模块）的 `build.gradle` 文件中添加依赖：

```gradle
dependencies {
    implementation project(':dehaze-sdk-android')
}
```

### 方式二：生成 AAR 文件并导入

1. 在 `dehaze-sdk-android` 目录下执行以下命令生成 AAR 文件：

```bash
./gradlew assembleRelease
```

2. 在你的 Android 项目中创建 `libs` 目录（如果不存在），将生成的 AAR 文件复制到该目录
3. 在需要使用 SDK 的模块（如 app 模块）的 `build.gradle` 文件中添加依赖：

```gradle
dependencies {
    implementation files('libs/dehaze-sdk-android-release.aar')
    // 注意：还需要手动添加 SDK 的依赖项
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.9.1'
    compileOnly 'org.projectlombok:lombok:1.18.22'
    annotationProcessor 'org.projectlombok:lombok:1.18.22'
}
```

## 初始化

在 Application 的 onCreate 方法中初始化 SDK：

```java
public class MyApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        
        // 初始化 Dehaze SDK
        DehazeSDK.initialize(new DehazeSDK.Builder()
                .setBaseUrl("https://api.dehaze.com/")
                .setDebug(BuildConfig.DEBUG));
    }
}
```

## 使用示例

### 用户登录和Token管理

```java
// 创建登录请求对象
LoginRequest request = new LoginRequest();
request.setUsername("username");
request.setPassword("password");
request.setCaptchaCode("captcha");
request.setCaptchaKey("key");

// 执行登录
AuthAPI.login(request, new ApiCallback<LoginResponse>() {
    @Override
    public void onSuccess(LoginResponse data) {
        // 登录成功处理
        String token = data.getToken();
        // 保存Token供后续请求使用
        TokenManager.setToken(token);
    }

    @Override
    public void onError(int code, String message) {
        // 业务错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
        Toast.makeText(context, "登录失败: " + friendlyMessage, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
        Toast.makeText(context, "网络错误: " + friendlyMessage, Toast.LENGTH_SHORT).show();
    }
});
```

### 获取验证码

```java
// 获取验证码
AuthAPI.getCaptcha(new ApiCallback<CaptchaResponse>() {
    @Override
    public void onSuccess(CaptchaResponse data) {
        // 获取验证码成功，显示验证码图片
        String captchaKey = data.getCaptchaKey();
        String captchaBase64 = data.getCaptchaBase64();
    }

    @Override
    public void onError(int code, String message) {
        // 业务错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 获取用户信息

```java
// 获取当前用户信息
UserAPI.getCurrentUserInfo(new ApiCallback<UserInfoResponse>() {
    @Override
    public void onSuccess(UserInfoResponse data) {
        // 获取用户信息成功
        String username = data.getUsername();
        String nickname = data.getNickname();
    }

    @Override
    public void onError(int code, String message) {
        // 业务错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 用户管理

```java
// 获取用户分页列表
UserQuery query = new UserQuery();
query.setPageNum(1);
query.setPageSize(10);
query.setKeywords("test");
UserAPI.getPage(query, new ApiCallback<PageResult<UserPageVO>>() {
    @Override
    public void onSuccess(PageResult<UserPageVO> data) {
        // 处理用户分页数据
        List<UserPageVO> users = data.getList();
        long total = data.getTotal();
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});

// 添加用户
UserForm userForm = new UserForm();
userForm.setUsername("newuser");
userForm.setNickname("New User");
// 设置其他属性...
UserAPI.add(userForm, new ApiCallback<Void>() {
    @Override
    public void onSuccess(Void data) {
        // 添加成功
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 算法相关API

```java
// 获取算法列表
AlgorithmQuery query = new AlgorithmQuery();
query.setKeywords("test");
AlgorithmAPI.getList(query, new ApiCallback<List<Algorithm>>() {
    @Override
    public void onSuccess(List<Algorithm> data) {
        // 处理算法列表
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 数据集相关API

```java
// 获取数据集列表
DatasetQuery query = new DatasetQuery();
query.setKeywords("test");
DatasetAPI.getList(query, new ApiCallback<List<Dataset>>() {
    @Override
    public void onSuccess(List<Dataset> data) {
        // 处理数据集列表
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 部门相关API

```java
// 获取部门列表
DeptQuery query = new DeptQuery();
query.setKeywords("test");
DeptAPI.getList(query, new ApiCallback<List<DeptVO>>() {
    @Override
    public void onSuccess(List<DeptVO> data) {
        // 处理部门列表
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 字典相关API

```java
// 获取字典类型分页列表
DictTypeQuery query = new DictTypeQuery();
query.setPageNum(1);
query.setPageSize(10);
query.setKeywords("test");
DictAPI.getDictTypePage(query, new ApiCallback<PageResult<DictTypePageVO>>() {
    @Override
    public void onSuccess(PageResult<DictTypePageVO> data) {
        // 处理字典类型分页数据
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 文件相关API

```java
// 上传文件
File file = new File("/path/to/file");
FileAPI.upload(file, null, new ApiCallback<FileInfo>() {
    @Override
    public void onSuccess(FileInfo data) {
        // 文件上传成功
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});

// 导出用户数据到文件
UserQuery query = new UserQuery();
query.setPageNum(1);
query.setPageSize(100);
UserAPI.export(query, "/sdcard/users_export.xlsx", new ApiCallback<Void>() {
    @Override
    public void onSuccess(Void data) {
        // 导出成功
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 菜单相关API

```java
// 获取菜单列表
MenuQuery query = new MenuQuery();
query.setKeywords("test");
MenuAPI.getList(query, new ApiCallback<List<MenuVO>>() {
    @Override
    public void onSuccess(List<MenuVO> data) {
        // 处理菜单列表
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 模型相关API

```java
// 模型预测
PredParam param = new PredParam();
param.setModelId(1);
param.setUrl("http://example.com/image.jpg");
ModelAPI.prediction(param, new ApiCallback<PredResult>() {
    @Override
    public void onSuccess(PredResult data) {
        // 处理预测结果
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

### 角色相关API

```java
// 获取角色分页列表
RoleQuery query = new RoleQuery();
query.setPageNum(1);
query.setPageSize(10);
query.setKeywords("test");
RoleAPI.getPage(query, new ApiCallback<PageResult<RolePageVO>>() {
    @Override
    public void onSuccess(PageResult<RolePageVO> data) {
        // 处理角色分页数据
    }

    @Override
    public void onError(int code, String message) {
        // 错误处理
        String friendlyMessage = ErrorUtils.parseError(new ApiException(code, message));
    }

    @Override
    public void onFailure(ApiException e) {
        // 网络错误处理
        String friendlyMessage = ErrorUtils.parseError(e);
    }
});
```

## API模块列表

- AuthAPI - 认证相关接口
- UserAPI - 用户相关接口
- AlgorithmAPI - 算法相关接口
- DatasetAPI - 数据集相关接口
- DeptAPI - 部门相关接口
- DictAPI - 字典相关接口
- FileAPI - 文件相关接口
- MenuAPI - 菜单相关接口
- ModelAPI - 模型相关接口
- RoleAPI - 角色相关接口

## 工具类

- TokenManager - Token管理工具
- ErrorUtils - 错误处理工具

## 测试

本SDK包含完整的测试套件，包括单元测试和集成测试。请参考 [TESTING.md](TESTING.md) 了解如何运行测试。

## 依赖库

- Retrofit2 (2.9.0)
- OkHttp3 (4.9.1)
- Gson (Retrofit converter)
- Lombok (1.18.22)
- Timber (5.0.1)

## 许可证

ISC
