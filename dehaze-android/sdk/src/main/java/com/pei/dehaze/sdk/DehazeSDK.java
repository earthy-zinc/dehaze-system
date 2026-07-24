package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.service.AlgorithmApiService;
import com.pei.dehaze.sdk.service.AlgorithmSelectApiService;
import com.pei.dehaze.sdk.service.ApiKeyApiService;
import com.pei.dehaze.sdk.service.AuthApiService;
import com.pei.dehaze.sdk.service.DatasetApiService;
import com.pei.dehaze.sdk.service.DeptApiService;
import com.pei.dehaze.sdk.service.DictApiService;
import com.pei.dehaze.sdk.service.FileApiService;
import com.pei.dehaze.sdk.service.InputHistoryApiService;
import com.pei.dehaze.sdk.service.MenuApiService;
import com.pei.dehaze.sdk.service.ModelApiService;
import com.pei.dehaze.sdk.service.RoleApiService;
import com.pei.dehaze.sdk.service.TaskApiService;
import com.pei.dehaze.sdk.service.UserApiService;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.input_history.InputSource;
import com.pei.dehaze.sdk.model.input_history.ProcessStatus;
import com.pei.dehaze.sdk.model.menu.MenuType;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.user.Gender;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;

import lombok.Getter;
import okhttp3.Authenticator;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.Route;
import okhttp3.logging.HttpLoggingInterceptor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * SDK主类，用于初始化和配置API客户端
 */
@Getter
public class DehazeSDK {
    private static volatile DehazeSDK instance;

    private final Retrofit retrofit;
    private final AuthApiService authApiService;
    private final ApiKeyApiService apiKeyApiService;
    private final UserApiService userApiService;
    private final AlgorithmApiService algorithmApiService;
    private final AlgorithmSelectApiService algorithmSelectApiService;
    private final DatasetApiService datasetApiService;
    private final DeptApiService deptApiService;
    private final DictApiService dictApiService;
    private final FileApiService fileApiService;
    private final InputHistoryApiService inputHistoryApiService;
    private final MenuApiService menuApiService;
    private final ModelApiService modelApiService;
    private final RoleApiService roleApiService;
    private final TaskApiService taskApiService;

    /** 公开接口路径（不需要 Token 注入） */
    private static final List<String> PUBLIC_ENDPOINTS = Arrays.asList(
            "/api/v1/auth/login",
            "/api/v1/auth/captcha"
    );

    /** Token 刷新同步锁，防止并发请求同时触发多次刷新 */
    private static final Object TOKEN_REFRESH_LOCK = new Object();

    private DehazeSDK(Builder builder) {
        String baseUrl = builder.baseUrl;

        // 配置OkHttp客户端
        OkHttpClient.Builder okHttpClientBuilder = new OkHttpClient.Builder();

        // 添加Token拦截器（公开接口跳过）
        okHttpClientBuilder.addInterceptor(new Interceptor() {
            @NotNull
            @Override
            public Response intercept(Chain chain) throws IOException {
                Request originalRequest = chain.request();
                Request.Builder requestBuilder = originalRequest.newBuilder()
                        .header("Accept", "application/json");

                // 仅对非公开接口注入 Token
                String path = originalRequest.url().encodedPath();
                boolean isPublic = false;
                for (String endpoint : PUBLIC_ENDPOINTS) {
                    if (path.endsWith(endpoint)) {
                        isPublic = true;
                        break;
                    }
                }

                if (!isPublic) {
                    String token = TokenManager.getToken();
                    if (token != null && !token.isEmpty()) {
                        requestBuilder.header("Authorization", "Bearer " + token);
                    }
                }

                Request newRequest = requestBuilder.build();
                return chain.proceed(newRequest);
            }
        });

        // 添加日志拦截器（通过 System.out 输出，Android 会重定向到 logcat，tag=System.out）
        if (builder.debug) {
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor(message ->
                    System.out.println("DehazeSDK " + message));
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
            okHttpClientBuilder.addInterceptor(loggingInterceptor);
        }

        // 添加 401 自动刷新 Token 的 Authenticator
        // 触发条件：accessToken 过期/无效导致后端返回 401
        // 行为：使用 refreshToken 同步刷新，成功后用新 accessToken 重放原始请求；失败则清除全部 token
        okHttpClientBuilder.authenticator(new Authenticator() {
            @Nullable
            @Override
            public Request authenticate(@Nullable Route route, @NotNull Response response) throws IOException {
                // 避免无限重试：已重试过的请求不再刷新
                if (response.request().header(HEADER_TOKEN_RETRIED) != null) {
                    return null;
                }
                String path = response.request().url().encodedPath();
                // 刷新接口自身的 401 不再递归刷新（refreshToken 已失效）
                if (path != null && path.endsWith("/api/v1/auth/refresh")) {
                    TokenManager.clearAll();
                    return null;
                }
                String refreshToken = TokenManager.getRefreshToken();
                if (refreshToken == null || refreshToken.isEmpty()) {
                    return null;
                }
                synchronized (TOKEN_REFRESH_LOCK) {
                    // 并发场景：其他线程可能已完成刷新。比较失败请求携带的 token 与当前 token，
                    // 若不同说明已被刷新，直接用新 token 重放
                    String failedToken = extractBearerToken(response.request());
                    String currentToken = TokenManager.getToken();
                    if (failedToken != null && !failedToken.equals(currentToken) && currentToken != null) {
                        return response.request().newBuilder()
                                .header("Authorization", "Bearer " + currentToken)
                                .header(HEADER_TOKEN_RETRIED, "1")
                                .build();
                    }
                    // 执行同步刷新
                    if (refreshTokenSynchronously(refreshToken)) {
                        String newToken = TokenManager.getToken();
                        return response.request().newBuilder()
                                .header("Authorization", "Bearer " + newToken)
                                .header(HEADER_TOKEN_RETRIED, "1")
                                .build();
                    }
                    // 刷新失败，清除全部 token，交由上层 ApiCallback 处理
                    TokenManager.clearAll();
                    return null;
                }
            }
        });

        // 构建Retrofit实例
        // 配置 Gson 日期反序列化：兼容后端多种日期格式
        // - "yyyy-MM-dd HH:mm:ss" 标准日期时间
        // - "yyyy-MM-dd HH:mm" 无秒级精度
        // - "yyyy-MM-dd" 仅日期（部分表/字段无时分秒）
        // - ISO 8601 "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"（部分接口返回）
        String[] dateFormats = {
                "yyyy-MM-dd HH:mm:ss",
                "yyyy-MM-dd HH:mm",
                "yyyy-MM-dd",
                "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
                "yyyy-MM-dd'T'HH:mm:ss'Z'"
        };
        JsonDeserializer<Date> dateDeserializer = (json, typeOfT, context) -> {
            if (json.isJsonNull()) return null;
            String dateStr = json.getAsString();
            if (dateStr == null || dateStr.isEmpty()) return null;
            for (String pattern : dateFormats) {
                try {
                    return new SimpleDateFormat(pattern, java.util.Locale.CHINA).parse(dateStr);
                } catch (ParseException ignored) {
                }
            }
            return null;
        };
        GsonBuilder gsonBuilder = new GsonBuilder()
                .registerTypeAdapter(Date.class, dateDeserializer)
                // AlgorithmStatus: 按 int value 序列化/反序列化
                .registerTypeAdapter(AlgorithmStatus.class,
                        (JsonDeserializer<AlgorithmStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : AlgorithmStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(AlgorithmStatus.class,
                        (JsonSerializer<AlgorithmStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // TaskStatus: 按 String value 序列化/反序列化
                .registerTypeAdapter(TaskStatus.class,
                        (JsonDeserializer<TaskStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : TaskStatus.fromValue(json.getAsString()))
                .registerTypeAdapter(TaskStatus.class,
                        (JsonSerializer<TaskStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // TaskType: 按 String value 序列化/反序列化
                .registerTypeAdapter(TaskType.class,
                        (JsonDeserializer<TaskType>) (json, type, ctx) ->
                                json.isJsonNull() ? null : TaskType.fromValue(json.getAsString()))
                .registerTypeAdapter(TaskType.class,
                        (JsonSerializer<TaskType>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // EnableStatus: 按 int value 序列化/反序列化
                .registerTypeAdapter(EnableStatus.class,
                        (JsonDeserializer<EnableStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : EnableStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(EnableStatus.class,
                        (JsonSerializer<EnableStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // ProcessStatus: 按 int value 序列化/反序列化
                .registerTypeAdapter(ProcessStatus.class,
                        (JsonDeserializer<ProcessStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : ProcessStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(ProcessStatus.class,
                        (JsonSerializer<ProcessStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // Gender: 按 int value 序列化/反序列化
                .registerTypeAdapter(Gender.class,
                        (JsonDeserializer<Gender>) (json, type, ctx) ->
                                json.isJsonNull() ? null : Gender.fromValue(json.getAsInt()))
                .registerTypeAdapter(Gender.class,
                        (JsonSerializer<Gender>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // InputSource: 按 String value 序列化/反序列化
                .registerTypeAdapter(InputSource.class,
                        (JsonDeserializer<InputSource>) (json, type, ctx) ->
                                json.isJsonNull() ? null : InputSource.fromValue(json.getAsString()))
                .registerTypeAdapter(InputSource.class,
                        (JsonSerializer<InputSource>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                // MenuType: 按 String value 序列化/反序列化
                .registerTypeAdapter(MenuType.class,
                        (JsonDeserializer<MenuType>) (json, type, ctx) ->
                                json.isJsonNull() ? null : MenuType.fromValue(json.getAsString()))
                .registerTypeAdapter(MenuType.class,
                        (JsonSerializer<MenuType>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()));
        retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(okHttpClientBuilder.build())
                .addConverterFactory(GsonConverterFactory.create(gsonBuilder.create()))
                .build();

        // 创建API服务实例
        this.authApiService = retrofit.create(AuthApiService.class);
        this.apiKeyApiService = retrofit.create(ApiKeyApiService.class);
        this.userApiService = retrofit.create(UserApiService.class);
        this.algorithmApiService = retrofit.create(AlgorithmApiService.class);
        this.algorithmSelectApiService = retrofit.create(AlgorithmSelectApiService.class);
        this.datasetApiService = retrofit.create(DatasetApiService.class);
        this.deptApiService = retrofit.create(DeptApiService.class);
        this.dictApiService = retrofit.create(DictApiService.class);
        this.fileApiService = retrofit.create(FileApiService.class);
        this.inputHistoryApiService = retrofit.create(InputHistoryApiService.class);
        this.menuApiService = retrofit.create(MenuApiService.class);
        this.modelApiService = retrofit.create(ModelApiService.class);
        this.roleApiService = retrofit.create(RoleApiService.class);
        this.taskApiService = retrofit.create(TaskApiService.class);
    }

    public static DehazeSDK getInstance() {
        if (instance == null) {
            synchronized (DehazeSDK.class) {
                if (instance == null) {
                    throw new IllegalStateException("DehazeSDK未初始化，请先调用initialize方法");
                }
            }
        }
        return instance;
    }

    /**
     * 将后端返回的相对 URL 解析为绝对 URL。
     * - 以 http:// 或 https:// 开头：直接返回
     * - 以 / 开头：拼接 baseUrl
     * - 其他：原样返回（可能是本地文件路径或已是绝对路径）
     */
    public String resolveUrl(String url) {
        if (url == null || url.isEmpty()) {
            return url;
        }
        if (url.startsWith("http://") || url.startsWith("https://")) {
            return url;
        }
        if (url.startsWith("/")) {
            return retrofit.baseUrl().toString().replaceAll("/+$", "") + url;
        }
        return url;
    }

    /**
     * 同步刷新 Token：调用 /api/v1/auth/refresh，成功后更新 TokenManager 中的 accessToken 与 refreshToken。
     * 调用方需自行加锁（{@link #TOKEN_REFRESH_LOCK}）以避免并发刷新。
     *
     * @param refreshToken 当前 refreshToken
     * @return true 表示刷新成功；false 表示刷新失败（refreshToken 已失效或网络错误）
     */
    private boolean refreshTokenSynchronously(String refreshToken) {
        try {
            retrofit2.Response<Result<com.pei.dehaze.sdk.model.auth.LoginResponse>> response =
                    authApiService.refreshToken(refreshToken).execute();
            if (response.isSuccessful() && response.body() != null) {
                Result<com.pei.dehaze.sdk.model.auth.LoginResponse> result = response.body();
                if (result.isSuccess() && result.getData() != null) {
                    com.pei.dehaze.sdk.model.auth.LoginResponse data = result.getData();
                    TokenManager.setToken(data.getAccessToken());
                    TokenManager.setRefreshToken(data.getRefreshToken());
                    return true;
                }
            }
            return false;
        } catch (IOException e) {
            return false;
        }
    }

    /** 标记已尝试 Token 刷新重放的请求头，避免 Authenticator 无限递归 */
    private static final String HEADER_TOKEN_RETRIED = "X-Token-Retried";

    /**
     * 从请求的 Authorization 头中提取 Bearer token 值
     */
    private static String extractBearerToken(Request request) {
        String header = request.header("Authorization");
        if (header != null && header.startsWith("Bearer ")) {
            return header.substring(7);
        }
        return null;
    }

    public static void initialize(Builder builder) {
        synchronized (DehazeSDK.class) {
            instance = new DehazeSDK(builder);
        }
    }

    public static class Builder {
        private String baseUrl = "";
        private boolean debug = false;

        public Builder setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder setDebug(boolean debug) {
            this.debug = debug;
            return this;
        }
    }
}
