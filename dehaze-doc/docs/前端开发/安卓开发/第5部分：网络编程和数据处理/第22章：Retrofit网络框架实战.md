# 第22章：Retrofit网络框架实战

## 🏗️ Retrofit架构和原理

### Retrofit核心架构

Retrofit是Square公司开发的类型安全的HTTP客户端，它通过Java接口定义REST API，使用注解描述HTTP请求，并与OkHttp无缝集成，提供了简洁而强大的网络请求解决方案。

```java
// RetrofitArchitecture.java
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

import java.util.List;
import java.util.concurrent.TimeUnit;

public class RetrofitArchitecture {

    // 1. Retrofit的核心组件
    public static class RetrofitComponents {
        // Retrofit实例 - 核心调度器
        private Retrofit retrofit;

        // Service接口 - API定义
        private ApiService apiService;

        // Call对象 - 请求执行器
        private Call<ApiResponse> call;

        // Converter - 数据转换器
        private GsonConverterFactory converterFactory;

        // OkHttp客户端 - 网络执行器
        private okhttp3.OkHttpClient okHttpClient;
    }

    // 2. Retrofit工作流程图解
    /*
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   API Interface │───▶│    Retrofit     │───▶│   Call Adapter   │
    │   定义接口方法    │    │   构建请求对象    │    │   适配调用类型    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                    │                          │
                                    ▼                          ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Call Factory  │◀───│   Converter     │    │   Call Object    │
    │   创建HTTP调用   │    │   数据类型转换    │    │   执行网络请求    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                    │                          │
                                    ▼                          ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   OkHttp Client │◀───│   Interceptors  │    │   Response       │
    │   执行网络请求    │    │   请求响应拦截    │    │   处理响应结果    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
    */

    // 3. 数据转换流程
    public static class DataConversionFlow {
        /*
        请求流程:
        Java对象 → Converter序列化 → RequestBody → OkHttp请求 → 网络传输

        响应流程:
        网络响应 → ResponseBody → Converter反序列化 → Java对象
        */
    }

    public static void demonstrateRetrofitArchitecture() {
        // 1. 创建OkHttp客户端
        okhttp3.OkHttpClient okHttpClient = new okhttp3.OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .addInterceptor(new LoggingInterceptor())
                .build();

        // 2. 创建Retrofit实例
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://api.example.com/")
                .client(okHttpClient) // 设置OkHttp客户端
                .addConverterFactory(GsonConverterFactory.create()) // 添加转换器
                .build();

        // 3. 创建API服务
        ApiService apiService = retrofit.create(ApiService.class);

        // 4. 执行网络请求
        Call<ApiResponse> call = apiService.getData();
        call.enqueue(new Callback<ApiResponse>() {
            @Override
            public void onResponse(Call<ApiResponse> call, Response<ApiResponse> response) {
                if (response.isSuccessful()) {
                    ApiResponse data = response.body();
                    System.out.println("请求成功: " + data);
                } else {
                    System.out.println("请求失败: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<ApiResponse> call, Throwable t) {
                System.out.println("网络错误: " + t.getMessage());
            }
        });
    }

    // 日志拦截器示例
    private static class LoggingInterceptor implements okhttp3.Interceptor {
        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            okhttp3.Request request = chain.request();
            System.out.println("发送请求: " + request.method() + " " + request.url());

            long startTime = System.currentTimeMillis();
            okhttp3.Response response = chain.proceed(request);
            long duration = System.currentTimeMillis() - startTime;

            System.out.println("收到响应: " + response.code() + " (" + duration + "ms)");
            return response;
        }
    }

    // API服务接口示例
    public interface ApiService {
        @GET("data")
        Call<ApiResponse> getData();
    }

    // API响应数据模型
    public static class ApiResponse {
        private int code;
        private String message;
        private Object data;

        // Getters and Setters
        public int getCode() { return code; }
        public void setCode(int code) { this.code = code; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public Object getData() { return data; }
        public void setData(Object data) { this.data = data; }

        @Override
        public String toString() {
            return String.format("ApiResponse{code=%d, message='%s', data=%s}", code, message, data);
        }
    }
}
```

## 🔌 API接口定义

### RESTful API接口设计

```java
// ApiInterfaceDefinition.java
import retrofit2.Call;
import retrofit2.http.*;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.Single;

import java.util.List;
import java.util.Map;

public class ApiInterfaceDefinition {

    // 1. 用户API接口
    public interface UserApiService {

        // GET请求 - 获取用户列表
        @GET("users")
        Call<List<User>> getUsers();

        // GET请求 - 带查询参数
        @GET("users")
        Call<List<User>> getUsers(@Query("page") int page, @Query("size") int size);

        // GET请求 - 路径参数
        @GET("users/{id}")
        Call<User> getUser(@Path("id") int userId);

        // GET请求 - 动态URL
        @GET
        Call<User> getUser(@Url String url);

        // POST请求 - 创建用户
        @POST("users")
        Call<User> createUser(@Body User user);

        // PUT请求 - 更新用户
        @PUT("users/{id}")
        Call<User> updateUser(@Path("id") int userId, @Body User user);

        // PATCH请求 - 部分更新用户
        @PATCH("users/{id}")
        Call<User> patchUser(@Path("id") int userId, @Body Map<String, Object> updates);

        // DELETE请求 - 删除用户
        @DELETE("users/{id}")
        Call<Void> deleteUser(@Path("id") int userId);

        // POST请求 - 表单数据
        @FormUrlEncoded
        @POST("users/login")
        Call<LoginResponse> login(@Field("username") String username,
                                  @Field("password") String password);

        // POST请求 - 多部分表单
        @Multipart
        @POST("users/avatar")
        Call<User> uploadAvatar(@Part("avatar") okhttp3.MultipartBody.Part avatar,
                               @Part("user_id") okhttp3.RequestBody userId);

        // GET请求 - Map查询参数
        @GET("users/search")
        Call<List<User>> searchUsers(@QueryMap Map<String, String> options);

        // HEAD请求 - 检查资源是否存在
        @HEAD("users/{id}")
        Call<Void> checkUserExists(@Path("id") int userId);
    }

    // 2. 文件上传API接口
    public interface FileApiService {

        // 单文件上传
        @Multipart
        @POST("upload")
        Call<UploadResponse> uploadFile(@Part("file") okhttp3.MultipartBody.Part file,
                                      @Part("description") okhttp3.RequestBody description);

        // 多文件上传
        @Multipart
        @POST("upload/multiple")
        Call<UploadResponse> uploadMultipleFiles(@Part List<okhttp3.MultipartBody.Part> files);

        // 文件下载
        @Streaming
        @GET("download/{filename}")
        Call<okhttp3.ResponseBody> downloadFile(@Path("filename") String filename);

        // 获取文件上传进度
        @GET("upload/progress/{taskId}")
        Call<UploadProgress> getUploadProgress(@Path("taskId") String taskId);
    }

    // 3. RxJava支持
    public interface RxUserApiService {

        // 返回Observable
        @GET("users")
        Observable<List<User>> getUsersObservable();

        // 返回Single
        @GET("users/{id}")
        Single<User> getUserSingle(@Path("id") int userId);

        // 返回Maybe
        @GET("users/{id}")
        io.reactivex.rxjava3.core.Maybe<User> getUserMaybe(@Path("id") int userId);
    }

    // 4. 数据模型类
    public static class User {
        private int id;
        private String username;
        private String email;
        private String avatar;
        private boolean active;
        private long createdAt;

        public User() {}

        public User(int id, String username, String email) {
            this.id = id;
            this.username = username;
            this.email = email;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }
        public long getCreatedAt() { return createdAt; }
        public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }

        @Override
        public String toString() {
            return String.format("User{id=%d, username='%s', email='%s', active=%s}",
                id, username, email, active);
        }
    }

    public static class LoginResponse {
        private String token;
        private User user;
        private long expiresIn;

        // Getters and Setters
        public String getToken() { return token; }
        public void setToken(String token) { this.token = token; }
        public User getUser() { return user; }
        public void setUser(User user) { this.user = user; }
        public long getExpiresIn() { return expiresIn; }
        public void setExpiresIn(long expiresIn) { this.expiresIn = expiresIn; }

        @Override
        public String toString() {
            return String.format("LoginResponse{token='%s', user=%s, expiresIn=%d}",
                token, user, expiresIn);
        }
    }

    public static class UploadResponse {
        private boolean success;
        private String message;
        private String fileId;
        private String url;

        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getFileId() { return fileId; }
        public void setFileId(String fileId) { this.fileId = fileId; }
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }

        @Override
        public String toString() {
            return String.format("UploadResponse{success=%s, message='%s', fileId='%s', url='%s'}",
                success, message, fileId, url);
        }
    }

    public static class UploadProgress {
        private String taskId;
        private long totalBytes;
        private long uploadedBytes;
        private double percentage;
        private String status; // "uploading", "completed", "failed"

        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public long getTotalBytes() { return totalBytes; }
        public void setTotalBytes(long totalBytes) { this.totalBytes = totalBytes; }
        public long getUploadedBytes() { return uploadedBytes; }
        public void setUploadedBytes(long uploadedBytes) { this.uploadedBytes = uploadedBytes; }
        public double getPercentage() { return percentage; }
        public void setPercentage(double percentage) { this.percentage = percentage; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }

        @Override
        public String toString() {
            return String.format("UploadProgress{taskId='%s', percentage=%.2f%%, status='%s'}",
                taskId, percentage, status);
        }
    }

    // 5. 注解使用示例
    public static class AnnotationExamples {

        // HTTP方法注解
        /*
        @GET    - GET请求
        @POST   - POST请求
        @PUT    - PUT请求
        @DELETE - DELETE请求
        @PATCH  - PATCH请求
        @HEAD   - HEAD请求
        @OPTIONS- OPTIONS请求
        */

        // 参数注解
        /*
        @Path   - 路径参数，替换URL中的占位符
        @Query  - 查询参数，添加到URL的查询字符串
        @QueryMap - 多个查询参数
        @Body   - 请求体，POST/PUT请求的数据
        @Field  - 表单字段
        @FieldMap- 多个表单字段
        @Part   - 多部分表单的一部分
        @PartMap- 多个多部分表单
        @Header - 请求头
        @HeaderMap- 多个请求头
        @Url    - 动态URL
        */

        // 标记注解
        /*
        @FormUrlEncoded - 表单编码
        @Multipart - 多部分表单
        @Streaming - 流式响应，用于大文件下载
        */
    }
}
```

## 🛡️ 拦截器和适配器

### 自定义拦截器

```java
// InterceptorAndAdapter.java
import android.util.Log;

import okhttp3.Interceptor;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.MediaType;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;

import retrofit2.Retrofit;
import retrofit2.adapter.rxjava3.RxJava3CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class InterceptorAndAdapter {

    // 1. 认证拦截器
    public static class AuthInterceptor implements Interceptor {
        private String token;

        public AuthInterceptor(String token) {
            this.token = token;
        }

        public void updateToken(String token) {
            this.token = token;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            Request originalRequest = chain.request();

            // 如果已有token，添加到请求头
            if (token != null && !token.isEmpty()) {
                Request authenticatedRequest = originalRequest.newBuilder()
                        .header("Authorization", "Bearer " + token)
                        .header("Accept", "application/json")
                        .build();

                Response response = chain.proceed(authenticatedRequest);

                // 检查是否需要刷新token
                if (response.code() == 401) {
                    // token过期，尝试刷新
                    String newToken = refreshToken();
                    if (newToken != null) {
                        // 使用新token重试请求
                        updateToken(newToken);
                        Request retryRequest = originalRequest.newBuilder()
                                .header("Authorization", "Bearer " + newToken)
                                .header("Accept", "application/json")
                                .build();
                        response.close(); // 关闭之前的响应
                        return chain.proceed(retryRequest);
                    }
                }

                return response;
            }

            return chain.proceed(originalRequest);
        }

        private String refreshToken() {
            // 实现token刷新逻辑
            try {
                // 这里应该调用刷新token的API
                Thread.sleep(1000); // 模拟网络延迟
                return "new_token_" + System.currentTimeMillis();
            } catch (Exception e) {
                Log.e("AuthInterceptor", "刷新token失败", e);
                return null;
            }
        }
    }

    // 2. 缓存拦截器
    public static class CacheInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            Response response = chain.proceed(request);

            // 为成功的响应添加缓存控制
            if (response.isSuccessful()) {
                String cacheControl = response.header("Cache-Control");
                if (cacheControl == null || cacheControl.contains("no-store") ||
                    cacheControl.contains("no-cache")) {
                    // 不缓存的响应
                    return response.newBuilder()
                            .header("Cache-Control", "public, max-age=60") // 缓存1分钟
                            .build();
                }
            }

            return response;
        }
    }

    // 3. 重试拦截器
    public static class RetryInterceptor implements Interceptor {
        private final int maxRetries;
        private final long retryDelay;

        public RetryInterceptor(int maxRetries, long retryDelay) {
            this.maxRetries = maxRetries;
            this.retryDelay = retryDelay;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            Response response = null;
            IOException lastException = null;

            for (int attempt = 0; attempt <= maxRetries; attempt++) {
                try {
                    if (attempt > 0) {
                        // 等待重试延迟
                        Thread.sleep(retryDelay * attempt);
                        Log.d("RetryInterceptor", String.format("重试第%d次", attempt));
                    }

                    response = chain.proceed(request);

                    // 如果响应成功，直接返回
                    if (response.isSuccessful()) {
                        return response;
                    }

                    // 某些状态码不应该重试
                    int code = response.code();
                    if (code == 400 || code == 401 || code == 403 || code == 404) {
                        return response;
                    }

                } catch (IOException e) {
                    lastException = e;
                    Log.w("RetryInterceptor", String.format("请求失败，第%d次尝试: %s",
                        attempt + 1, e.getMessage()));

                    if (response != null) {
                        response.close();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("重试被中断", e);
                }
            }

            // 所有重试都失败了
            if (lastException != null) {
                throw lastException;
            }

            return response != null ? response : chain.proceed(request);
        }
    }

    // 4. 响应拦截器
    public static class ResponseInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            long startTime = System.currentTimeMillis();

            Response response = chain.proceed(request);
            long duration = System.currentTimeMillis() - startTime;

            // 记录请求信息
            logRequest(request, response, duration);

            // 检查响应内容
            String responseBody = response.body().string();

            // 记录响应内容（可选）
            logResponse(responseBody);

            // 创建新的响应体
            MediaType contentType = response.body().contentType();
            ResponseBody newResponseBody = ResponseBody.create(contentType, responseBody);

            return response.newBuilder()
                    .body(newResponseBody)
                    .build();
        }

        private void logRequest(Request request, Response response, long duration) {
            Log.d("ResponseInterceptor", String.format(
                "请求: %s %s -> 响应: %d (%dms)",
                request.method(),
                request.url(),
                response.code(),
                duration
            ));
        }

        private void logResponse(String responseBody) {
            // 只记录前500个字符，避免日志过长
            if (responseBody.length() > 500) {
                Log.d("ResponseInterceptor", "响应内容: " + responseBody.substring(0, 500) + "...");
            } else {
                Log.d("ResponseInterceptor", "响应内容: " + responseBody);
            }
        }
    }

    // 5. 网络状态拦截器
    public static class NetworkInterceptor implements Interceptor {
        private final NetworkManager networkManager;

        public NetworkInterceptor(NetworkManager networkManager) {
            this.networkManager = networkManager;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            if (!networkManager.isNetworkAvailable()) {
                throw new IOException("网络不可用");
            }

            return chain.proceed(chain.request());
        }
    }

    // 网络管理器接口
    public interface NetworkManager {
        boolean isNetworkAvailable();
    }

    // 6. Retrofit配置工厂
    public static class RetrofitFactory {

        // 创建基础Retrofit实例
        public static Retrofit createRetrofit(String baseUrl) {
            return createRetrofit(baseUrl, null);
        }

        // 创建带认证的Retrofit实例
        public static Retrofit createRetrofit(String baseUrl, String token) {
            okhttp3.OkHttpClient.Builder clientBuilder = new okhttp3.OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(30, TimeUnit.SECONDS);

            // 添加日志拦截器
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor();
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
            clientBuilder.addInterceptor(loggingInterceptor);

            // 添加认证拦截器
            if (token != null) {
                clientBuilder.addInterceptor(new AuthInterceptor(token));
            }

            // 添加缓存拦截器
            clientBuilder.addInterceptor(new CacheInterceptor());

            // 添加重试拦截器
            clientBuilder.addInterceptor(new RetryInterceptor(3, 1000));

            // 添加响应拦截器
            clientBuilder.addInterceptor(new ResponseInterceptor());

            // 添加网络拦截器
            clientBuilder.addNetworkInterceptor(new NetworkInterceptor(() -> true));

            return new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(clientBuilder.build())
                    .addConverterFactory(GsonConverterFactory.create())
                    .addCallAdapterFactory(RxJava3CallAdapterFactory.create())
                    .build();
        }

        // 创建带所有拦截器的Retrofit实例
        public static Retrofit createFullRetrofit(String baseUrl, String token,
                                                 NetworkManager networkManager) {
            okhttp3.OkHttpClient.Builder clientBuilder = new okhttp3.OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(30, TimeUnit.SECONDS);

            // 按顺序添加拦截器
            clientBuilder.addInterceptor(new RetryInterceptor(3, 1000));

            if (token != null) {
                clientBuilder.addInterceptor(new AuthInterceptor(token));
            }

            clientBuilder.addInterceptor(new NetworkInterceptor(networkManager));
            clientBuilder.addInterceptor(new CacheInterceptor());
            clientBuilder.addInterceptor(new ResponseInterceptor());

            // 网络拦截器
            clientBuilder.addNetworkInterceptor(new HttpLoggingInterceptor()
                    .setLevel(HttpLoggingInterceptor.Level.HEADERS));

            okhttp3.OkHttpClient client = clientBuilder.build();

            return new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(client)
                    .addConverterFactory(GsonConverterFactory.create())
                    .addCallAdapterFactory(RxJava3CallAdapterFactory.create())
                    .build();
        }
    }

    // 7. 自定义CallAdapter
    public static class CustomCallAdapterFactory extends retrofit2.CallAdapter.Factory {
        @Override
        public retrofit2.CallAdapter<?, ?> get(Type returnType, Annotation[] annotations,
                                             retrofit2.Retrofit retrofit) {
            // 检查返回类型是否是CustomCall
            if (getRawType(returnType) == CustomCall.class) {
                Type responseType = getParameterUpperBound(0, (ParameterizedType) returnType);
                return new CustomCallAdapter(responseType);
            }
            return null;
        }

        private static class CustomCallAdapter implements retrofit2.CallAdapter<Object, CustomCall<?>> {
            private final Type responseType;

            CustomCallAdapter(Type responseType) {
                this.responseType = responseType;
            }

            @Override
            public Type responseType() {
                return responseType;
            }

            @Override
            public CustomCall<?> adapt(retrofit2.Call<Object> call) {
                return new CustomCall<>(call);
            }
        }
    }

    // 自定义Call包装类
    public static class CustomCall<T> {
        private final retrofit2.Call<T> call;

        public CustomCall(retrofit2.Call<T> call) {
            this.call = call;
        }

        public void enqueue(CustomCallback<T> callback) {
            call.enqueue(new retrofit2.Callback<T>() {
                @Override
                public void onResponse(retrofit2.Call<T> call, retrofit2.Response<T> response) {
                    if (response.isSuccessful()) {
                        callback.onSuccess(response.body());
                    } else {
                        callback.onError(new Exception("HTTP " + response.code()));
                    }
                }

                @Override
                public void onFailure(retrofit2.Call<T> call, Throwable t) {
                    callback.onError(t);
                }
            });
        }
    }

    // 自定义回调接口
    public interface CustomCallback<T> {
        void onSuccess(T result);
        void onError(Throwable error);
    }

    // 使用示例
    public static void demonstrateInterceptors() {
        // 创建Retrofit实例
        Retrofit retrofit = RetrofitFactory.createFullRetrofit(
            "https://api.example.com/",
            "your_token_here",
            () -> true // 模拟网络可用
        );

        // 创建API服务
        ApiInterfaceDefinition.UserApiService apiService =
            retrofit.create(ApiInterfaceDefinition.UserApiService.class);

        // 执行请求
        apiService.getUsers().enqueue(new retrofit2.Callback<List<ApiInterfaceDefinition.User>>() {
            @Override
            public void onResponse(retrofit2.Call<List<ApiInterfaceDefinition.User>> call,
                                   retrofit2.Response<List<ApiInterfaceDefinition.User>> response) {
                if (response.isSuccessful()) {
                    List<ApiInterfaceDefinition.User> users = response.body();
                    System.out.println("获取用户列表成功: " + users);
                } else {
                    System.out.println("获取用户列表失败: " + response.code());
                }
            }

            @Override
            public void onFailure(retrofit2.Call<List<ApiInterfaceDefinition.User>> call, Throwable t) {
                System.out.println("网络请求失败: " + t.getMessage());
            }
        });
    }
}
```

## 📁 文件上传实现

### 单文件和多文件上传

```java
// FileUploadImplementation.java
import android.util.Log;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class FileUploadImplementation {

    // 1. 文件上传服务接口
    public interface FileUploadService {

        // 单文件上传
        @Multipart
        @POST("upload/single")
        Call<UploadResponse> uploadSingleFile(@Part MultipartBody.Part file,
                                            @Part("description") RequestBody description,
                                            @Part("user_id") RequestBody userId);

        // 多文件上传
        @Multipart
        @POST("upload/multiple")
        Call<UploadResponse> uploadMultipleFiles(@Part List<MultipartBody.Part> files,
                                               @Part("user_id") RequestBody userId);

        // 带进度监听的上传
        @Multipart
        @POST("upload/progress")
        Call<UploadResponse> uploadWithProgress(@Part MultipartBody.Part file,
                                              @Part("task_id") RequestBody taskId);

        // 分块上传
        @Multipart
        @POST("upload/chunk")
        Call<ChunkUploadResponse> uploadChunk(@Part("file") MultipartBody.Part chunk,
                                            @Part("chunk_index") RequestBody chunkIndex,
                                            @Part("total_chunks") RequestBody totalChunks,
                                            @Part("file_id") RequestBody fileId,
                                            @Part("file_name") RequestBody fileName);

        // 完成分块上传
        @POST("upload/complete")
        Call<UploadResponse> completeChunkUpload(@Body CompleteUploadRequest request);

        // 文件下载
        @Streaming
        @GET("download/{fileId}")
        Call<ResponseBody> downloadFile(@Path("fileId") String fileId);

        // 获取上传进度
        @GET("upload/progress/{taskId}")
        Call<UploadProgress> getUploadProgress(@Path("taskId") String taskId);
    }

    // 2. 文件上传管理器
    public static class FileUploadManager {
        private static final String TAG = "FileUploadManager";
        private final FileUploadService uploadService;
        private final Map<String, UploadTask> uploadTasks;
        private final OkHttpClient httpClient;

        public FileUploadManager(String baseUrl) {
            this.httpClient = new OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .writeTimeout(60, TimeUnit.SECONDS)
                    .build();

            Retrofit retrofit = new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(httpClient)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();

            this.uploadService = retrofit.create(FileUploadService.class);
            this.uploadTasks = new HashMap<>();
        }

        // 3. 单文件上传
        public void uploadSingleFile(File file, String description, String userId,
                                    UploadCallback callback) {
            if (!file.exists()) {
                callback.onError(new Exception("文件不存在"));
                return;
            }

            try {
                // 创建文件请求体
                RequestBody fileBody = RequestBody.create(
                    MediaType.parse(getMimeType(file)), file);

                // 创建多部分体
                MultipartBody.Part filePart = MultipartBody.Part.createFormData(
                    "file", file.getName(), fileBody);

                RequestBody descriptionBody = RequestBody.create(
                    MediaType.parse("text/plain"), description);

                RequestBody userIdBody = RequestBody.create(
                    MediaType.parse("text/plain"), userId);

                // 创建上传任务
                String taskId = generateTaskId();
                UploadTask task = new UploadTask(taskId, file.getName(), file.length());
                uploadTasks.put(taskId, task);

                // 执行上传
                Call<UploadResponse> call = uploadService.uploadSingleFile(
                    filePart, descriptionBody, userIdBody);

                call.enqueue(new Callback<UploadResponse>() {
                    @Override
                    public void onResponse(Call<UploadResponse> call, Response<UploadResponse> response) {
                        if (response.isSuccessful()) {
                            UploadResponse uploadResponse = response.body();
                            task.setStatus(UploadStatus.COMPLETED);
                            task.setProgress(100);
                            callback.onSuccess(uploadResponse);
                        } else {
                            task.setStatus(UploadStatus.FAILED);
                            callback.onError(new Exception("上传失败: " + response.code()));
                        }
                        uploadTasks.remove(taskId);
                    }

                    @Override
                    public void onFailure(Call<UploadResponse> call, Throwable t) {
                        task.setStatus(UploadStatus.FAILED);
                        callback.onError(t);
                        uploadTasks.remove(taskId);
                    }
                });

            } catch (Exception e) {
                callback.onError(e);
            }
        }

        // 4. 多文件上传
        public void uploadMultipleFiles(List<File> files, String userId,
                                       MultiUploadCallback callback) {
            if (files == null || files.isEmpty()) {
                callback.onError(new Exception("文件列表为空"));
                return;
            }

            try {
                List<MultipartBody.Part> fileParts = new ArrayList<>();
                long totalSize = 0;

                for (File file : files) {
                    if (file.exists()) {
                        RequestBody fileBody = RequestBody.create(
                            MediaType.parse(getMimeType(file)), file);

                        MultipartBody.Part filePart = MultipartBody.Part.createFormData(
                            "files", file.getName(), fileBody);

                        fileParts.add(filePart);
                        totalSize += file.length();
                    }
                }

                if (fileParts.isEmpty()) {
                    callback.onError(new Exception("没有有效的文件"));
                    return;
                }

                RequestBody userIdBody = RequestBody.create(
                    MediaType.parse("text/plain"), userId);

                // 创建上传任务
                String taskId = generateTaskId();
                UploadTask task = new UploadTask(taskId, "多文件上传", totalSize);
                uploadTasks.put(taskId, task);

                Call<UploadResponse> call = uploadService.uploadMultipleFiles(fileParts, userIdBody);
                call.enqueue(new Callback<UploadResponse>() {
                    @Override
                    public void onResponse(Call<UploadResponse> call, Response<UploadResponse> response) {
                        if (response.isSuccessful()) {
                            UploadResponse uploadResponse = response.body();
                            task.setStatus(UploadStatus.COMPLETED);
                            task.setProgress(100);
                            callback.onSuccess(uploadResponse);
                        } else {
                            task.setStatus(UploadStatus.FAILED);
                            callback.onError(new Exception("多文件上传失败: " + response.code()));
                        }
                        uploadTasks.remove(taskId);
                    }

                    @Override
                    public void onFailure(Call<UploadResponse> call, Throwable t) {
                        task.setStatus(UploadStatus.FAILED);
                        callback.onError(t);
                        uploadTasks.remove(taskId);
                    }
                });

            } catch (Exception e) {
                callback.onError(e);
            }
        }

        // 5. 分块上传
        public void uploadFileInChunks(File file, String userId, ChunkUploadCallback callback) {
            final long CHUNK_SIZE = 1024 * 1024; // 1MB分块
            final String fileId = generateFileId();
            final String taskId = generateTaskId();

            new Thread(() -> {
                try {
                    long fileSize = file.length();
                    int totalChunks = (int) Math.ceil((double) fileSize / CHUNK_SIZE);

                    UploadTask task = new UploadTask(taskId, file.getName(), fileSize);
                    uploadTasks.put(taskId, task);

                    for (int chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                        long start = chunkIndex * CHUNK_SIZE;
                        long end = Math.min(start + CHUNK_SIZE, fileSize);
                        byte[] chunkData = readFileChunk(file, start, end);

                        // 创建分块请求体
                        RequestBody chunkBody = RequestBody.create(
                            MediaType.parse("application/octet-stream"), chunkData);

                        MultipartBody.Part chunkPart = MultipartBody.Part.createFormData(
                            "file", "chunk_" + chunkIndex, chunkBody);

                        RequestBody chunkIndexBody = RequestBody.create(
                            MediaType.parse("text/plain"), String.valueOf(chunkIndex));

                        RequestBody totalChunksBody = RequestBody.create(
                            MediaType.parse("text/plain"), String.valueOf(totalChunks));

                        RequestBody fileIdBody = RequestBody.create(
                            MediaType.parse("text/plain"), fileId);

                        RequestBody fileNameBody = RequestBody.create(
                            MediaType.parse("text/plain"), file.getName());

                        // 同步上传分块
                        Call<ChunkUploadResponse> call = uploadService.uploadChunk(
                            chunkPart, chunkIndexBody, totalChunksBody, fileIdBody, fileNameBody);

                        Response<ChunkUploadResponse> response = call.execute();

                        if (!response.isSuccessful()) {
                            throw new Exception("分块上传失败: " + response.code());
                        }

                        // 更新进度
                        double progress = (double) (chunkIndex + 1) / totalChunks * 100;
                        task.setProgress(progress);
                        callback.onChunkUploaded(chunkIndex + 1, totalChunks, progress);
                    }

                    // 完成分块上传
                    CompleteUploadRequest completeRequest = new CompleteUploadRequest(
                        fileId, file.getName(), totalChunks, userId);

                    Call<UploadResponse> completeCall = uploadService.completeChunkUpload(completeRequest);
                    Response<UploadResponse> completeResponse = completeCall.execute();

                    if (completeResponse.isSuccessful()) {
                        task.setStatus(UploadStatus.COMPLETED);
                        task.setProgress(100);
                        callback.onSuccess(completeResponse.body());
                    } else {
                        throw new Exception("完成上传失败: " + completeResponse.code());
                    }

                } catch (Exception e) {
                    UploadTask task = uploadTasks.get(taskId);
                    if (task != null) {
                        task.setStatus(UploadStatus.FAILED);
                    }
                    callback.onError(e);
                } finally {
                    uploadTasks.remove(taskId);
                }
            }).start();
        }

        // 6. 文件下载
        public void downloadFile(String fileId, String savePath, DownloadCallback callback) {
            Call<ResponseBody> call = uploadService.downloadFile(fileId);
            call.enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                    if (response.isSuccessful()) {
                        try {
                            saveDownloadedFile(response.body(), savePath, callback);
                        } catch (Exception e) {
                            callback.onError(e);
                        }
                    } else {
                        callback.onError(new Exception("下载失败: " + response.code()));
                    }
                }

                @Override
                public void onFailure(Call<ResponseBody> call, Throwable t) {
                    callback.onError(t);
                }
            });
        }

        // 保存下载的文件
        private void saveDownloadedFile(ResponseBody body, String savePath, DownloadCallback callback) {
            InputStream inputStream = null;
            FileOutputStream outputStream = null;

            try {
                byte[] buffer = new byte[4096];
                long fileSize = body.contentLength();
                long downloadedSize = 0;
                int bytesRead;

                inputStream = body.byteStream();
                outputStream = new FileOutputStream(savePath);

                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                    downloadedSize += bytesRead;

                    // 更新下载进度
                    if (fileSize > 0) {
                        double progress = (double) downloadedSize / fileSize * 100;
                        callback.onProgress(downloadedSize, fileSize, progress);
                    }
                }

                outputStream.flush();
                callback.onSuccess(new File(savePath));

            } catch (Exception e) {
                callback.onError(e);
            } finally {
                try {
                    if (inputStream != null) inputStream.close();
                    if (outputStream != null) outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭文件流失败", e);
                }
            }
        }

        // 获取文件MIME类型
        private String getMimeType(File file) {
            String fileName = file.getName().toLowerCase();
            if (fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
                return "image/jpeg";
            } else if (fileName.endsWith(".png")) {
                return "image/png";
            } else if (fileName.endsWith(".gif")) {
                return "image/gif";
            } else if (fileName.endsWith(".pdf")) {
                return "application/pdf";
            } else if (fileName.endsWith(".txt")) {
                return "text/plain";
            } else if (fileName.endsWith(".json")) {
                return "application/json";
            } else if (fileName.endsWith(".mp4")) {
                return "video/mp4";
            } else if (fileName.endsWith(".mp3")) {
                return "audio/mpeg";
            } else {
                return "application/octet-stream";
            }
        }

        // 读取文件分块
        private byte[] readFileChunk(File file, long start, long end) throws IOException {
            try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(file, "r")) {
                raf.seek(start);
                int length = (int) (end - start);
                byte[] chunk = new byte[length];
                raf.readFully(chunk);
                return chunk;
            }
        }

        // 生成任务ID
        private String generateTaskId() {
            return "task_" + System.currentTimeMillis() + "_" + (int)(Math.random() * 1000);
        }

        // 生成文件ID
        private String generateFileId() {
            return "file_" + System.currentTimeMillis() + "_" + (int)(Math.random() * 1000);
        }

        // 获取上传任务
        public UploadTask getUploadTask(String taskId) {
            return uploadTasks.get(taskId);
        }

        // 取消上传
        public void cancelUpload(String taskId) {
            UploadTask task = uploadTasks.remove(taskId);
            if (task != null) {
                task.setStatus(UploadStatus.CANCELLED);
            }
        }
    }

    // 7. 数据模型类
    public enum UploadStatus {
        PENDING, UPLOADING, COMPLETED, FAILED, CANCELLED
    }

    public static class UploadTask {
        private String taskId;
        private String fileName;
        private long totalSize;
        private double progress;
        private UploadStatus status;
        private long startTime;

        public UploadTask(String taskId, String fileName, long totalSize) {
            this.taskId = taskId;
            this.fileName = fileName;
            this.totalSize = totalSize;
            this.progress = 0;
            this.status = UploadStatus.PENDING;
            this.startTime = System.currentTimeMillis();
        }

        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public long getTotalSize() { return totalSize; }
        public void setTotalSize(long totalSize) { this.totalSize = totalSize; }
        public double getProgress() { return progress; }
        public void setProgress(double progress) { this.progress = progress; }
        public UploadStatus getStatus() { return status; }
        public void setStatus(UploadStatus status) { this.status = status; }
        public long getStartTime() { return startTime; }
        public void setStartTime(long startTime) { this.startTime = startTime; }

        @Override
        public String toString() {
            return String.format("UploadTask{taskId='%s', fileName='%s', progress=%.2f%%, status=%s}",
                taskId, fileName, progress, status);
        }
    }

    public static class UploadResponse {
        private boolean success;
        private String message;
        private String fileId;
        private String url;
        private long fileSize;

        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getFileId() { return fileId; }
        public void setFileId(String fileId) { this.fileId = fileId; }
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public long getFileSize() { return fileSize; }
        public void setFileSize(long fileSize) { this.fileSize = fileSize; }

        @Override
        public String toString() {
            return String.format("UploadResponse{success=%s, fileId='%s', url='%s', fileSize=%d}",
                success, fileId, url, fileSize);
        }
    }

    public static class ChunkUploadResponse {
        private boolean success;
        private int chunkIndex;
        private String message;

        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public int getChunkIndex() { return chunkIndex; }
        public void setChunkIndex(int chunkIndex) { this.chunkIndex = chunkIndex; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    public static class CompleteUploadRequest {
        private String fileId;
        private String fileName;
        private int totalChunks;
        private String userId;

        public CompleteUploadRequest(String fileId, String fileName, int totalChunks, String userId) {
            this.fileId = fileId;
            this.fileName = fileName;
            this.totalChunks = totalChunks;
            this.userId = userId;
        }

        // Getters and Setters
        public String getFileId() { return fileId; }
        public void setFileId(String fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public int getTotalChunks() { return totalChunks; }
        public void setTotalChunks(int totalChunks) { this.totalChunks = totalChunks; }
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
    }

    public static class UploadProgress {
        private String taskId;
        private long uploadedBytes;
        private long totalBytes;
        private double percentage;
        private UploadStatus status;

        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public long getUploadedBytes() { return uploadedBytes; }
        public void setUploadedBytes(long uploadedBytes) { this.uploadedBytes = uploadedBytes; }
        public long getTotalBytes() { return totalBytes; }
        public void setTotalBytes(long totalBytes) { this.totalBytes = totalBytes; }
        public double getPercentage() { return percentage; }
        public void setPercentage(double percentage) { this.percentage = percentage; }
        public UploadStatus getStatus() { return status; }
        public void setStatus(UploadStatus status) { this.status = status; }
    }

    // 8. 回调接口
    public interface UploadCallback {
        void onSuccess(UploadResponse response);
        void onError(Exception error);
    }

    public interface MultiUploadCallback {
        void onSuccess(UploadResponse response);
        void onError(Exception error);
    }

    public interface ChunkUploadCallback {
        void onChunkUploaded(int chunkNumber, int totalChunks, double progress);
        void onSuccess(UploadResponse response);
        void onError(Exception error);
    }

    public interface DownloadCallback {
        void onProgress(long downloaded, long total, double percentage);
        void onSuccess(File file);
        void onError(Exception error);
    }

    // 9. 使用示例
    public static void demonstrateFileUpload() {
        // 创建文件上传管理器
        FileUploadManager uploadManager = new FileUploadManager("https://api.example.com/");

        // 单文件上传
        File file = new File("/path/to/file.jpg");
        uploadManager.uploadSingleFile(file, "测试图片", "user123", new UploadCallback() {
            @Override
            public void onSuccess(UploadResponse response) {
                System.out.println("单文件上传成功: " + response);
            }

            @Override
            public void onError(Exception error) {
                System.err.println("单文件上传失败: " + error.getMessage());
            }
        });

        // 多文件上传
        List<File> files = List.of(
            new File("/path/to/file1.jpg"),
            new File("/path/to/file2.png")
        );
        uploadManager.uploadMultipleFiles(files, "user123", new MultiUploadCallback() {
            @Override
            public void onSuccess(UploadResponse response) {
                System.out.println("多文件上传成功: " + response);
            }

            @Override
            public void onError(Exception error) {
                System.err.println("多文件上传失败: " + error.getMessage());
            }
        });

        // 分块上传（适用于大文件）
        File largeFile = new File("/path/to/large_file.mp4");
        uploadManager.uploadFileInChunks(largeFile, "user123", new ChunkUploadCallback() {
            @Override
            public void onChunkUploaded(int chunkNumber, int totalChunks, double progress) {
                System.out.printf("分块上传进度: %d/%d (%.2f%%)%n",
                    chunkNumber, totalChunks, progress);
            }

            @Override
            public void onSuccess(UploadResponse response) {
                System.out.println("分块上传完成: " + response);
            }

            @Override
            public void onError(Exception error) {
                System.err.println("分块上传失败: " + error.getMessage());
            }
        });
    }
}
```

## 🚨 错误处理和重试机制

### 综合错误处理策略

```java
// ErrorHandlingAndRetry.java
import android.util.Log;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.HttpException;

import java.io.IOException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

public class ErrorHandlingAndRetry {

    // 1. 自定义异常类型
    public static class NetworkException extends Exception {
        private final ErrorType errorType;
        private final int statusCode;

        public NetworkException(ErrorType errorType, String message, int statusCode) {
            super(message);
            this.errorType = errorType;
            this.statusCode = statusCode;
        }

        public NetworkException(ErrorType errorType, String message) {
            this(errorType, message, -1);
        }

        public ErrorType getErrorType() { return errorType; }
        public int getStatusCode() { return statusCode; }

        @Override
        public String toString() {
            return String.format("NetworkException{type=%s, message='%s', statusCode=%d}",
                errorType, getMessage(), statusCode);
        }
    }

    public enum ErrorType {
        NETWORK_UNAVAILABLE,    // 网络不可用
        TIMEOUT,                // 超时
        SERVER_ERROR,           // 服务器错误
        CLIENT_ERROR,           // 客户端错误
        PARSE_ERROR,            // 解析错误
        UNKNOWN_ERROR           // 未知错误
    }

    // 2. 错误分析器
    public static class ErrorAnalyzer {
        private static final String TAG = "ErrorAnalyzer";

        public static NetworkException analyzeError(Throwable throwable) {
            Log.e(TAG, "分析网络错误", throwable);

            if (throwable instanceof NetworkException) {
                return (NetworkException) throwable;
            }

            if (throwable instanceof HttpException) {
                HttpException httpException = (HttpException) throwable;
                int code = httpException.code();

                if (code >= 500) {
                    return new NetworkException(ErrorType.SERVER_ERROR,
                        String.format("服务器错误: %d", code), code);
                } else if (code >= 400) {
                    return new NetworkException(ErrorType.CLIENT_ERROR,
                        String.format("客户端错误: %d", code), code);
                }
            }

            if (throwable instanceof SocketTimeoutException) {
                return new NetworkException(ErrorType.TIMEOUT, "请求超时");
            }

            if (throwable instanceof UnknownHostException) {
                return new NetworkException(ErrorType.NETWORK_UNAVAILABLE, "主机未知");
            }

            if (throwable instanceof IOException) {
                String message = throwable.getMessage();
                if (message != null && message.contains("Network")) {
                    return new NetworkException(ErrorType.NETWORK_UNAVAILABLE, "网络不可用");
                }
                return new NetworkException(ErrorType.UNKNOWN_ERROR, "IO错误: " + message);
            }

            return new NetworkException(ErrorType.UNKNOWN_ERROR,
                "未知错误: " + throwable.getMessage());
        }

        public static boolean isRetryableError(NetworkException error) {
            switch (error.getErrorType()) {
                case NETWORK_UNAVAILABLE:
                case TIMEOUT:
                case SERVER_ERROR:
                    return true;
                case CLIENT_ERROR:
                    return error.getStatusCode() == 408 || // Request Timeout
                           error.getStatusCode() == 429;    // Too Many Requests
                case PARSE_ERROR:
                case UNKNOWN_ERROR:
                default:
                    return false;
            }
        }

        public static long getRetryDelay(int retryCount, ErrorType errorType) {
            // 基础延迟时间
            long baseDelay = 1000; // 1秒

            switch (errorType) {
                case NETWORK_UNAVAILABLE:
                    return baseDelay * 2 * (retryCount + 1); // 递增延迟
                case TIMEOUT:
                    return baseDelay * (retryCount + 1);
                case SERVER_ERROR:
                    return baseDelay * 3 * (retryCount + 1); // 服务器错误延迟更长
                default:
                    return baseDelay;
            }
        }
    }

    // 3. 重试配置
    public static class RetryConfig {
        private final int maxRetries;
        private final long baseDelay;
        private final double backoffMultiplier;
        private final long maxDelay;
        private final boolean enableJitter;

        public RetryConfig(int maxRetries, long baseDelay, double backoffMultiplier,
                          long maxDelay, boolean enableJitter) {
            this.maxRetries = maxRetries;
            this.baseDelay = baseDelay;
            this.backoffMultiplier = backoffMultiplier;
            this.maxDelay = maxDelay;
            this.enableJitter = enableJitter;
        }

        // 预定义配置
        public static RetryConfig getConservativeConfig() {
            return new RetryConfig(3, 1000, 2.0, 10000, true);
        }

        public static RetryConfig getAggressiveConfig() {
            return new RetryConfig(5, 500, 1.5, 5000, true);
        }

        public static RetryConfig getQuickRetryConfig() {
            return new RetryConfig(2, 200, 1.0, 1000, false);
        }

        // 计算重试延迟
        public long calculateDelay(int retryCount) {
            long delay = (long) (baseDelay * Math.pow(backoffMultiplier, retryCount));
            delay = Math.min(delay, maxDelay);

            if (enableJitter) {
                // 添加±25%的随机抖动
                double jitter = 0.5 + Math.random(); // 0.5-1.5
                delay = (long) (delay * jitter);
            }

            return delay;
        }

        // Getters
        public int getMaxRetries() { return maxRetries; }
        public long getBaseDelay() { return baseDelay; }
        public double getBackoffMultiplier() { return backoffMultiplier; }
        public long getMaxDelay() { return maxDelay; }
        public boolean isEnableJitter() { return enableJitter; }
    }

    // 4. 重试执行器
    public static class RetryExecutor {
        private static final String TAG = "RetryExecutor";

        public static <T> void executeWithRetry(Call<T> call, RetryConfig config,
                                               RetryCallback<T> callback) {
            executeWithRetry(call, config, callback, 0);
        }

        private static <T> void executeWithRetry(Call<T> call, RetryConfig config,
                                                RetryCallback<T> callback, int retryCount) {
            Log.d(TAG, String.format("执行请求，重试次数: %d/%d", retryCount, config.getMaxRetries()));

            call.enqueue(new Callback<T>() {
                @Override
                public void onResponse(Call<T> call, Response<T> response) {
                    if (response.isSuccessful()) {
                        callback.onSuccess(response.body());
                    } else {
                        NetworkException error = new NetworkException(
                            ErrorType.CLIENT_ERROR,
                            "HTTP错误: " + response.code(),
                            response.code()
                        );

                        handleRetry(call, config, callback, error, retryCount);
                    }
                }

                @Override
                public void onFailure(Call<T> call, Throwable t) {
                    NetworkException error = ErrorAnalyzer.analyzeError(t);
                    handleRetry(call, config, callback, error, retryCount);
                }
            });
        }

        private static <T> void handleRetry(Call<T> originalCall, RetryConfig config,
                                           RetryCallback<T> callback, NetworkException error,
                                           int retryCount) {
            Log.w(TAG, String.format("请求失败: %s, 重试次数: %d", error, retryCount));

            if (retryCount < config.getMaxRetries() && ErrorAnalyzer.isRetryableError(error)) {
                long delay = config.calculateDelay(retryCount);
                Log.d(TAG, String.format("将在%dms后重试", delay));

                // 延迟后重试
                android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
                mainHandler.postDelayed(() -> {
                    // 克隆call以重新执行
                    Call<T> newCall = originalCall.clone();
                    executeWithRetry(newCall, config, callback, retryCount + 1);
                }, delay);
            } else {
                Log.e(TAG, "重试次数已用尽或错误不可重试");
                callback.onError(error);
            }
        }
    }

    // 5. 带重试的Retrofit调用包装器
    public static class RetrofitCallWrapper {
        private final RetryConfig defaultRetryConfig;

        public RetrofitCallWrapper() {
            this.defaultRetryConfig = RetryConfig.getConservativeConfig();
        }

        public RetrofitCallWrapper(RetryConfig defaultRetryConfig) {
            this.defaultRetryConfig = defaultRetryConfig;
        }

        // 执行带重试的同步调用
        public <T> T executeSync(Call<T> call) throws NetworkException {
            return executeSync(call, defaultRetryConfig);
        }

        public <T> T executeSync(Call<T> call, RetryConfig config) throws NetworkException {
            NetworkException lastError = null;

            for (int retryCount = 0; retryCount <= config.getMaxRetries(); retryCount++) {
                try {
                    if (retryCount > 0) {
                        long delay = config.calculateDelay(retryCount - 1);
                        Thread.sleep(delay);
                    }

                    Response<T> response = call.execute();
                    if (response.isSuccessful()) {
                        return response.body();
                    } else {
                        lastError = new NetworkException(
                            ErrorType.CLIENT_ERROR,
                            "HTTP错误: " + response.code(),
                            response.code()
                        );

                        if (!ErrorAnalyzer.isRetryableError(lastError) || retryCount == config.getMaxRetries()) {
                            throw lastError;
                        }
                    }

                } catch (IOException e) {
                    lastError = ErrorAnalyzer.analyzeError(e);
                    if (!ErrorAnalyzer.isRetryableError(lastError) || retryCount == config.getMaxRetries()) {
                        throw lastError;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new NetworkException(ErrorType.UNKNOWN_ERROR, "重试被中断");
                }
            }

            throw lastError != null ? lastError :
                new NetworkException(ErrorType.UNKNOWN_ERROR, "重试失败");
        }

        // 执行带重试的异步调用
        public <T> void executeAsync(Call<T> call, RetryCallback<T> callback) {
            executeAsync(call, defaultRetryConfig, callback);
        }

        public <T> void executeAsync(Call<T> call, RetryConfig config, RetryCallback<T> callback) {
            RetryExecutor.executeWithRetry(call, config, callback);
        }
    }

    // 6. 回调接口
    public interface RetryCallback<T> {
        void onSuccess(T result);
        void onError(NetworkException error);
    }

    // 7. 网络状态监听器
    public static class NetworkStateListener {
        private boolean isNetworkAvailable = true;
        private NetworkStateCallback callback;

        public interface NetworkStateCallback {
            void onNetworkAvailable();
            void onNetworkUnavailable();
        }

        public void setCallback(NetworkStateCallback callback) {
            this.callback = callback;
        }

        public void updateNetworkState(boolean isAvailable) {
            if (isNetworkAvailable != isAvailable) {
                isNetworkAvailable = isAvailable;
                if (callback != null) {
                    if (isAvailable) {
                        callback.onNetworkAvailable();
                    } else {
                        callback.onNetworkUnavailable();
                    }
                }
            }
        }

        public boolean isNetworkAvailable() {
            return isNetworkAvailable;
        }
    }

    // 8. 错误处理统计
    public static class ErrorStatistics {
        private long totalRequests;
        private long successfulRequests;
        private long failedRequests;
        private final Map<ErrorType, Long> errorCounts;
        private final Map<Integer, Long> statusCodeCounts;

        public ErrorStatistics() {
            this.errorCounts = new java.util.concurrent.ConcurrentHashMap<>();
            this.statusCodeCounts = new java.util.concurrent.ConcurrentHashMap<>();
        }

        public void recordRequest(boolean success, NetworkException error) {
            totalRequests++;
            if (success) {
                successfulRequests++;
            } else {
                failedRequests++;
                if (error != null) {
                    errorCounts.merge(error.getErrorType(), 1L, Long::sum);
                    if (error.getStatusCode() > 0) {
                        statusCodeCounts.merge(error.getStatusCode(), 1L, Long::sum);
                    }
                }
            }
        }

        public double getSuccessRate() {
            return totalRequests > 0 ? (double) successfulRequests / totalRequests * 100 : 0;
        }

        public String getStatisticsReport() {
            StringBuilder report = new StringBuilder();
            report.append(String.format("请求统计: 总数=%d, 成功=%d, 失败=%d, 成功率=%.2f%%\n",
                totalRequests, successfulRequests, failedRequests, getSuccessRate()));

            if (!errorCounts.isEmpty()) {
                report.append("错误类型统计:\n");
                errorCounts.forEach((type, count) ->
                    report.append(String.format("  %s: %d\n", type, count)));
            }

            if (!statusCodeCounts.isEmpty()) {
                report.append("状态码统计:\n");
                statusCodeCounts.forEach((code, count) ->
                    report.append(String.format("  %d: %d\n", code, count)));
            }

            return report.toString();
        }

        public void reset() {
            totalRequests = 0;
            successfulRequests = 0;
            failedRequests = 0;
            errorCounts.clear();
            statusCodeCounts.clear();
        }
    }

    // 9. 使用示例
    public static void demonstrateErrorHandling() {
        // 创建Retrofit调用包装器
        RetrofitCallWrapper wrapper = new RetrofitCallWrapper(RetryConfig.getAggressiveConfig());

        // 模拟API调用
        // Call<String> call = apiService.getData();

        // 异步调用带重试
        /*
        wrapper.executeAsync(call, new RetryCallback<String>() {
            @Override
            public void onSuccess(String result) {
                System.out.println("请求成功: " + result);
            }

            @Override
            public void onError(NetworkException error) {
                System.err.println("请求最终失败: " + error);
            }
        });
        */

        // 同步调用带重试
        /*
        try {
            String result = wrapper.executeSync(call);
            System.out.println("请求成功: " + result);
        } catch (NetworkException e) {
            System.err.println("请求最终失败: " + e);
        }
        */

        // 网络状态监听
        NetworkStateListener networkListener = new NetworkStateListener();
        networkListener.setCallback(new NetworkStateListener.NetworkStateCallback() {
            @Override
            public void onNetworkAvailable() {
                System.out.println("网络已连接，可以执行请求");
            }

            @Override
            public void onNetworkUnavailable() {
                System.out.println("网络不可用，暂停请求");
            }
        });

        // 错误统计
        ErrorStatistics statistics = new ErrorStatistics();
        // statistics.recordRequest(true, null); // 记录成功请求
        // statistics.recordRequest(false, error); // 记录失败请求

        System.out.println(statistics.getStatisticsReport());
    }
}
```

## 📱 实践示例：完整的网络客户端

### 综合网络客户端实现

```java
// NetworkClientExample.java
import android.content.Context;
import android.util.Log;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class NetworkClientExample {

    // 1. 完整的网络客户端
    public static class NetworkClient {
        private static final String TAG = "NetworkClient";
        private static NetworkClient instance;

        private final Retrofit retrofit;
        private final ApiService apiService;
        private final String baseUrl;
        private final String authToken;
        private final ErrorStatistics errorStatistics;

        private NetworkClient(Context context, String baseUrl, String authToken) {
            this.baseUrl = baseUrl;
            this.authToken = authToken;
            this.errorStatistics = new ErrorStatistics();

            // 创建Retrofit实例
            this.retrofit = createRetrofitInstance();
            this.apiService = retrofit.create(ApiService.class);
        }

        public static synchronized NetworkClient getInstance(Context context, String baseUrl, String authToken) {
            if (instance == null) {
                instance = new NetworkClient(context.getApplicationContext(), baseUrl, authToken);
            }
            return instance;
        }

        // 创建Retrofit实例
        private Retrofit createRetrofitInstance() {
            // 创建OkHttp客户端
            okhttp3.OkHttpClient client = new okhttp3.OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .writeTimeout(60, TimeUnit.SECONDS)
                    .addInterceptor(new AuthInterceptor(authToken))
                    .addInterceptor(new LoggingInterceptor())
                    .addInterceptor(new CacheInterceptor())
                    .addInterceptor(new ErrorHandlingInterceptor())
                    .addNetworkInterceptor(new NetworkInterceptor())
                    .build();

            return new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(client)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }

        // 用户相关API
        public void getUsers(PaginationCallback<User> callback) {
            Call<ApiResponse<List<User>>> call = apiService.getUsers();
            call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));
        }

        public void getUser(int userId, ResultCallback<User> callback) {
            Call<ApiResponse<User>> call = apiService.getUser(userId);
            call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));
        }

        public void createUser(User user, ResultCallback<User> callback) {
            Call<ApiResponse<User>> call = apiService.createUser(user);
            call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));
        }

        public void updateUser(int userId, User user, ResultCallback<User> callback) {
            Call<ApiResponse<User>> call = apiService.updateUser(userId, user);
            call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));
        }

        public void deleteUser(int userId, ResultCallback<Void> callback) {
            Call<ApiResponse<Void>> call = apiService.deleteUser(userId);
            call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));
        }

        // 文件上传
        public void uploadUserAvatar(int userId, java.io.File imageFile,
                                    ProgressCallback<UploadResponse> callback) {
            try {
                okhttp3.RequestBody requestFile = okhttp3.RequestBody.create(
                    okhttp3.MediaType.parse("image/*"), imageFile);

                MultipartBody.Part body = MultipartBody.Part.createFormData(
                    "avatar", imageFile.getName(), requestFile);

                okhttp3.RequestBody userIdBody = okhttp3.RequestBody.create(
                    okhttp3.MediaType.parse("text/plain"), String.valueOf(userId));

                Call<ApiResponse<UploadResponse>> call = apiService.uploadAvatar(userIdBody, body);
                call.enqueue(new ApiResponseCallback<>(callback, errorStatistics));

            } catch (Exception e) {
                callback.onError(new NetworkException(ErrorType.UNKNOWN_ERROR, e.getMessage()));
            }
        }

        // 获取错误统计
        public String getErrorStatistics() {
            return errorStatistics.getStatisticsReport();
        }

        // 重置统计
        public void resetStatistics() {
            errorStatistics.reset();
        }
    }

    // 2. API服务接口
    public interface ApiService {
        @GET("users")
        Call<ApiResponse<List<User>>> getUsers();

        @GET("users/{id}")
        Call<ApiResponse<User>> getUser(@Path("id") int id);

        @POST("users")
        Call<ApiResponse<User>> createUser(@Body User user);

        @PUT("users/{id}")
        Call<ApiResponse<User>> updateUser(@Path("id") int id, @Body User user);

        @DELETE("users/{id}")
        Call<ApiResponse<Void>> deleteUser(@Path("id") int id);

        @Multipart
        @POST("users/avatar")
        Call<ApiResponse<UploadResponse>> uploadAvatar(@Part("user_id") okhttp3.RequestBody userId,
                                                     @Part MultipartBody.Part avatar);
    }

    // 3. 通用API响应格式
    public static class ApiResponse<T> {
        private boolean success;
        private String message;
        private T data;
        private int code;
        private long timestamp;

        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public T getData() { return data; }
        public void setData(T data) { this.data = data; }
        public int getCode() { return code; }
        public void setCode(int code) { this.code = code; }
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }

    // 4. 数据模型
    public static class User {
        private int id;
        private String username;
        private String email;
        private String avatar;
        private boolean active;
        private long createdAt;

        public User() {}

        public User(int id, String username, String email) {
            this.id = id;
            this.username = username;
            this.email = email;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }
        public long getCreatedAt() { return createdAt; }
        public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }

        @Override
        public String toString() {
            return String.format("User{id=%d, username='%s', email='%s', active=%s}",
                id, username, email, active);
        }
    }

    public static class UploadResponse {
        private boolean success;
        private String message;
        private String url;
        private long size;

        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public long getSize() { return size; }
        public void setSize(long size) { this.size = size; }

        @Override
        public String toString() {
            return String.format("UploadResponse{success=%s, url='%s', size=%d}",
                success, url, size);
        }
    }

    // 5. 拦截器实现
    public static class AuthInterceptor implements okhttp3.Interceptor {
        private final String token;

        public AuthInterceptor(String token) {
            this.token = token;
        }

        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            okhttp3.Request originalRequest = chain.request();
            okhttp3.Request.Builder requestBuilder = originalRequest.newBuilder();

            if (token != null && !token.isEmpty()) {
                requestBuilder.header("Authorization", "Bearer " + token);
            }

            requestBuilder.header("Accept", "application/json");
            requestBuilder.header("User-Agent", "AndroidApp/1.0");

            return chain.proceed(requestBuilder.build());
        }
    }

    public static class LoggingInterceptor implements okhttp3.Interceptor {
        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            okhttp3.Request request = chain.request();
            long startTime = System.currentTimeMillis();

            Log.d(TAG, String.format("发送请求: %s %s", request.method(), request.url()));

            okhttp3.Response response = chain.proceed(request);
            long duration = System.currentTimeMillis() - startTime;

            Log.d(TAG, String.format("收到响应: %d (%dms)", response.code(), duration));

            return response;
        }
    }

    public static class CacheInterceptor implements okhttp3.Interceptor {
        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            okhttp3.Request request = chain.request();
            okhttp3.Response response = chain.proceed(request);

            if (response.isSuccessful()) {
                return response.newBuilder()
                        .header("Cache-Control", "public, max-age=300") // 缓存5分钟
                        .build();
            }

            return response;
        }
    }

    public static class ErrorHandlingInterceptor implements okhttp3.Interceptor {
        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            okhttp3.Request request = chain.request();
            okhttp3.Response response = chain.proceed(request);

            // 记录非成功响应
            if (!response.isSuccessful()) {
                Log.w(TAG, String.format("请求失败: %s %s -> %d",
                    request.method(), request.url(), response.code()));
            }

            return response;
        }
    }

    public static class NetworkInterceptor implements okhttp3.Interceptor {
        @Override
        public okhttp3.Response intercept(okhttp3.Interceptor.Chain chain) throws IOException {
            // 这里可以添加网络层面的处理逻辑
            return chain.proceed(chain.request());
        }
    }

    // 6. 回调接口
    public interface ResultCallback<T> {
        void onSuccess(T result);
        void onError(NetworkException error);
    }

    public interface PaginationCallback<T> {
        void onSuccess(List<T> results);
        void onError(NetworkException error);
    }

    public interface ProgressCallback<T> extends ResultCallback<T> {
        void onProgress(double progress);
    }

    // 7. API响应回调包装器
    private static class ApiResponseCallback<T> implements Callback<ApiResponse<T>> {
        private final ResultCallback<T> callback;
        private final ErrorStatistics errorStatistics;

        public ApiResponseCallback(ResultCallback<T> callback, ErrorStatistics errorStatistics) {
            this.callback = callback;
            this.errorStatistics = errorStatistics;
        }

        @Override
        public void onResponse(Call<ApiResponse<T>> call, Response<ApiResponse<T>> response) {
            try {
                if (response.isSuccessful()) {
                    ApiResponse<T> apiResponse = response.body();
                    if (apiResponse != null && apiResponse.isSuccess()) {
                        callback.onSuccess(apiResponse.getData());
                        errorStatistics.recordRequest(true, null);
                    } else {
                        String message = apiResponse != null ? apiResponse.getMessage() : "未知错误";
                        NetworkException error = new NetworkException(
                            ErrorType.CLIENT_ERROR, message, response.code());
                        callback.onError(error);
                        errorStatistics.recordRequest(false, error);
                    }
                } else {
                    NetworkException error = new NetworkException(
                        ErrorType.CLIENT_ERROR, "HTTP错误: " + response.code(), response.code());
                    callback.onError(error);
                    errorStatistics.recordRequest(false, error);
                }
            } catch (Exception e) {
                NetworkException error = new NetworkException(
                    ErrorType.PARSE_ERROR, "响应解析失败: " + e.getMessage());
                callback.onError(error);
                errorStatistics.recordRequest(false, error);
            }
        }

        @Override
        public void onFailure(Call<ApiResponse<T>> call, Throwable t) {
            NetworkException error = ErrorHandlingAndRetry.ErrorAnalyzer.analyzeError(t);
            callback.onError(error);
            errorStatistics.recordRequest(false, error);
        }
    }

    // 8. 网络异常类（复用之前的实现）
    public static class NetworkException extends Exception {
        private final ErrorHandlingAndRetry.ErrorType errorType;
        private final int statusCode;

        public NetworkException(ErrorHandlingAndRetry.ErrorType errorType, String message, int statusCode) {
            super(message);
            this.errorType = errorType;
            this.statusCode = statusCode;
        }

        public ErrorHandlingAndRetry.ErrorType getErrorType() { return errorType; }
        public int getStatusCode() { return statusCode; }

        @Override
        public String toString() {
            return String.format("NetworkException{type=%s, message='%s', statusCode=%d}",
                errorType, getMessage(), statusCode);
        }
    }

    // 9. 错误统计类（复用之前的实现）
    public static class ErrorStatistics {
        private long totalRequests;
        private long successfulRequests;
        private long failedRequests;
        private final Map<ErrorHandlingAndRetry.ErrorType, Long> errorCounts;
        private final Map<Integer, Long> statusCodeCounts;

        public ErrorStatistics() {
            this.errorCounts = new java.util.concurrent.ConcurrentHashMap<>();
            this.statusCodeCounts = new java.util.concurrent.ConcurrentHashMap<>();
        }

        public void recordRequest(boolean success, NetworkException error) {
            totalRequests++;
            if (success) {
                successfulRequests++;
            } else {
                failedRequests++;
                if (error != null) {
                    errorCounts.merge(error.getErrorType(), 1L, Long::sum);
                    if (error.getStatusCode() > 0) {
                        statusCodeCounts.merge(error.getStatusCode(), 1L, Long::sum);
                    }
                }
            }
        }

        public double getSuccessRate() {
            return totalRequests > 0 ? (double) successfulRequests / totalRequests * 100 : 0;
        }

        public String getStatisticsReport() {
            StringBuilder report = new StringBuilder();
            report.append(String.format("请求统计: 总数=%d, 成功=%d, 失败=%d, 成功率=%.2f%%\n",
                totalRequests, successfulRequests, failedRequests, getSuccessRate()));

            if (!errorCounts.isEmpty()) {
                report.append("错误类型统计:\n");
                errorCounts.forEach((type, count) ->
                    report.append(String.format("  %s: %d\n", type, count)));
            }

            if (!statusCodeCounts.isEmpty()) {
                report.append("状态码统计:\n");
                statusCodeCounts.forEach((code, count) ->
                    report.append(String.format("  %d: %d\n", code, count)));
            }

            return report.toString();
        }

        public void reset() {
            totalRequests = 0;
            successfulRequests = 0;
            failedRequests = 0;
            errorCounts.clear();
            statusCodeCounts.clear();
        }
    }

    // 10. 使用示例
    public static void demonstrateNetworkClient() {
        // 模拟Android Context
        Context mockContext = null; // 在实际使用中传入真实的Context

        if (mockContext != null) {
            // 创建网络客户端
            NetworkClient client = NetworkClient.getInstance(
                mockContext,
                "https://api.example.com/",
                "your_auth_token_here"
            );

            // 获取用户列表
            client.getUsers(new PaginationCallback<User>() {
                @Override
                public void onSuccess(List<User> users) {
                    System.out.println("获取用户列表成功: " + users);
                }

                @Override
                public void onError(NetworkException error) {
                    System.err.println("获取用户列表失败: " + error);
                }
            });

            // 获取单个用户
            client.getUser(1, new ResultCallback<User>() {
                @Override
                public void onSuccess(User user) {
                    System.out.println("获取用户成功: " + user);
                }

                @Override
                public void onError(NetworkException error) {
                    System.err.println("获取用户失败: " + error);
                }
            });

            // 创建用户
            User newUser = new User(0, "新用户", "newuser@example.com");
            client.createUser(newUser, new ResultCallback<User>() {
                @Override
                public void onSuccess(User user) {
                    System.out.println("创建用户成功: " + user);
                }

                @Override
                public void onError(NetworkException error) {
                    System.err.println("创建用户失败: " + error);
                }
            });

            // 打印错误统计
            System.out.println(client.getErrorStatistics());
        }
    }
}
```

## 📝 本章小结

### 核心知识点

1. **Retrofit架构和原理**
   - Retrofit的核心组件和工作流程
   - 数据转换机制和类型安全
   - 与OkHttp的集成关系
   - 设计模式和架构思想

2. **API接口定义**
   - RESTful API接口设计原则
   - HTTP方法注解的使用
   - 参数注解的灵活应用
   - 动态URL和查询参数处理

3. **拦截器和适配器**
   - 自定义拦截器的实现
   - 认证、缓存、重试拦截器
   - CallAdapter的工作原理
   - 响应处理和错误拦截

4. **文件上传实现**
   - 单文件和多文件上传
   - 分块上传大文件
   - 进度监听和状态管理
   - Multipart请求体构建

5. **错误处理和重试机制**
   - 异常类型分类和分析
   - 智能重试策略设计
   - 网络状态监听
   - 错误统计和分析

6. **综合网络客户端**
   - 统一的API调用接口
   - 模块化的架构设计
   - 完整的错误处理体系
   - 性能监控和统计

### 实践建议

1. **Retrofit使用**
   - 合理设计API接口
   - 使用注解简化代码
   - 配置合适的转换器
   - 实现统一的错误处理

2. **拦截器应用**
   - 按需添加拦截器
   - 注意拦截器执行顺序
   - 避免重复处理
   - 监控网络性能

3. **文件上传优化**
   - 大文件使用分块上传
   - 实现进度监听
   - 处理网络中断情况
   - 优化内存使用

4. **错误处理策略**
   - 分类处理不同错误
   - 实现智能重试
   - 提供用户友好的错误信息
   - 收集错误统计信息

### 常见问题解决

1. **网络请求失败**
   - 检查网络连接状态
   - 验证API接口定义
   - 确认服务器地址正确
   - 检查权限配置

2. **数据解析错误**
   - 确认数据模型匹配
   - 检查JSON格式
   - 验证转换器配置
   - 处理null值情况

3. **文件上传问题**
   - 检查文件路径和权限
   - 确认MIME类型正确
   - 处理大文件上传
   - 实现断点续传

4. **性能优化问题**
   - 使用连接池
   - 配置合适的超时时间
   - 实现请求缓存
   - 监控网络性能

通过本章的学习，你掌握了Retrofit网络框架的完整使用方法，包括架构理解、API设计、拦截器实现、文件上传、错误处理和综合网络客户端构建。这些技能为开发高质量的网络应用提供了全面的技术支持。Retrofit的类型安全特性和简洁的API设计，使其成为Android网络编程的首选框架。