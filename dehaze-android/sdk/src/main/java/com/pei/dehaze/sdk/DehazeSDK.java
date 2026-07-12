package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.service.AlgorithmApiService;
import com.pei.dehaze.sdk.service.AlgorithmSelectApiService;
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

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import lombok.Getter;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.logging.HttpLoggingInterceptor;
import org.jetbrains.annotations.NotNull;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * SDK主类，用于初始化和配置API客户端
 */
@Getter
public class DehazeSDK {
    private static volatile DehazeSDK instance;

    private final Retrofit retrofit;
    private final AuthApiService authApiService;
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

        // 添加日志拦截器
        if (builder.debug) {
            Logger logger = Logger.getLogger("DehazeSDK");
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor(logger::info);
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
            okHttpClientBuilder.addInterceptor(loggingInterceptor);
        }

        // 构建Retrofit实例
        retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(okHttpClientBuilder.build())
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        // 创建API服务实例
        this.authApiService = retrofit.create(AuthApiService.class);
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

        public DehazeSDK build() {
            return new DehazeSDK(this);
        }
    }
}
