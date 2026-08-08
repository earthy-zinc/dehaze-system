package com.pei.dehaze.sdk;

import com.pei.dehaze.sdk.service.AlgorithmApiService;
import com.pei.dehaze.sdk.service.AlgorithmSelectApiService;
import com.pei.dehaze.sdk.service.AnnouncementApiService;
import com.pei.dehaze.sdk.service.ApiKeyApiService;
import com.pei.dehaze.sdk.service.AuthApiService;
import com.pei.dehaze.sdk.service.DatasetApiService;
import com.pei.dehaze.sdk.service.DeptApiService;
import com.pei.dehaze.sdk.service.DictApiService;
import com.pei.dehaze.sdk.service.FavoriteApiService;
import com.pei.dehaze.sdk.service.FeedbackApiService;
import com.pei.dehaze.sdk.service.FileApiService;
import com.pei.dehaze.sdk.service.InputHistoryApiService;
import com.pei.dehaze.sdk.service.MemberApiService;
import com.pei.dehaze.sdk.service.MenuApiService;
import com.pei.dehaze.sdk.service.MessageApiService;
import com.pei.dehaze.sdk.service.MessageTemplateApiService;
import com.pei.dehaze.sdk.service.ModelApiService;
import com.pei.dehaze.sdk.service.NotificationSettingApiService;
import com.pei.dehaze.sdk.service.OrderApiService;
import com.pei.dehaze.sdk.service.PackageApiService;
import com.pei.dehaze.sdk.service.RecommendationApiService;
import com.pei.dehaze.sdk.service.RoleApiService;
import com.pei.dehaze.sdk.service.TaskApiService;
import com.pei.dehaze.sdk.service.UserApiService;
import com.pei.dehaze.sdk.network.TraceInterceptor;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.input_history.InputSource;
import com.pei.dehaze.sdk.model.input_history.ProcessStatus;
import com.pei.dehaze.sdk.model.menu.MenuType;
import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.user.Gender;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;

import lombok.Getter;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Response;
import okhttp3.logging.HttpLoggingInterceptor;
import org.jetbrains.annotations.NotNull;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

@Getter
public class DehazeSDK {
    private static volatile DehazeSDK instance;

    private final Retrofit retrofit;
    private final AuthApiService authApiService;
    private final ApiKeyApiService apiKeyApiService;
    private final UserApiService userApiService;
    private final AlgorithmApiService algorithmApiService;
    private final AlgorithmSelectApiService algorithmSelectApiService;
    private final FavoriteApiService favoriteApiService;
    private final RecommendationApiService recommendationApiService;
    private final DatasetApiService datasetApiService;
    private final DeptApiService deptApiService;
    private final DictApiService dictApiService;
    private final FileApiService fileApiService;
    private final InputHistoryApiService inputHistoryApiService;
    private final MenuApiService menuApiService;
    private final ModelApiService modelApiService;
    private final RoleApiService roleApiService;
    private final TaskApiService taskApiService;
    private final MemberApiService memberApiService;
    private final PackageApiService packageApiService;
    private final OrderApiService orderApiService;
    private final FeedbackApiService feedbackApiService;
    private final MessageApiService messageApiService;
    private final AnnouncementApiService announcementApiService;
    private final MessageTemplateApiService messageTemplateApiService;
    private final NotificationSettingApiService notificationSettingApiService;

    private static final List<String> PUBLIC_ENDPOINTS = Arrays.asList(
            "/api/v1/auth/login",
            "/api/v1/auth/captcha"
    );

    private DehazeSDK(Builder builder) {
        String baseUrl = builder.baseUrl;

        OkHttpClient.Builder okHttpClientBuilder = new OkHttpClient.Builder();

        // 日志与 trace_id 拦截器（注入 X-Trace-Id、失败上报）
        okHttpClientBuilder.addInterceptor(new TraceInterceptor());

        okHttpClientBuilder.addInterceptor(new Interceptor() {
            @NotNull
            @Override
            public Response intercept(Chain chain) throws IOException {
                okhttp3.Request originalRequest = chain.request();
                okhttp3.Request.Builder requestBuilder = originalRequest.newBuilder()
                        .header("Accept", "application/json");

                String path = originalRequest.url().encodedPath();
                boolean isPublic = false;
                for (String endpoint : PUBLIC_ENDPOINTS) {
                    if (path.endsWith(endpoint)) {
                        isPublic = true;
                        break;
                    }
                }

                if (!isPublic) {
                    String sid = TokenManager.getSessionId();
                    if (sid != null && !sid.isEmpty()) {
                        requestBuilder.header("X-Session-Id", sid);
                    }
                }

                return chain.proceed(requestBuilder.build());
            }
        });

        if (builder.debug) {
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor(message ->
                    System.out.println("DehazeSDK " + message));
            loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
            okHttpClientBuilder.addInterceptor(loggingInterceptor);
        }

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
                .registerTypeAdapter(AlgorithmStatus.class,
                        (JsonDeserializer<AlgorithmStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : AlgorithmStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(AlgorithmStatus.class,
                        (JsonSerializer<AlgorithmStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(TaskStatus.class,
                        (JsonDeserializer<TaskStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : TaskStatus.fromValue(json.getAsString()))
                .registerTypeAdapter(TaskStatus.class,
                        (JsonSerializer<TaskStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(TaskType.class,
                        (JsonDeserializer<TaskType>) (json, type, ctx) ->
                                json.isJsonNull() ? null : TaskType.fromValue(json.getAsString()))
                .registerTypeAdapter(TaskType.class,
                        (JsonSerializer<TaskType>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(EnableStatus.class,
                        (JsonDeserializer<EnableStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : EnableStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(EnableStatus.class,
                        (JsonSerializer<EnableStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(ProcessStatus.class,
                        (JsonDeserializer<ProcessStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : ProcessStatus.fromValue(json.getAsInt()))
                .registerTypeAdapter(ProcessStatus.class,
                        (JsonSerializer<ProcessStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(Gender.class,
                        (JsonDeserializer<Gender>) (json, type, ctx) ->
                                json.isJsonNull() ? null : Gender.fromValue(json.getAsInt()))
                .registerTypeAdapter(Gender.class,
                        (JsonSerializer<Gender>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(InputSource.class,
                        (JsonDeserializer<InputSource>) (json, type, ctx) ->
                                json.isJsonNull() ? null : InputSource.fromValue(json.getAsString()))
                .registerTypeAdapter(InputSource.class,
                        (JsonSerializer<InputSource>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(MenuType.class,
                        (JsonDeserializer<MenuType>) (json, type, ctx) ->
                                json.isJsonNull() ? null : MenuType.fromValue(json.getAsString()))
                .registerTypeAdapter(MenuType.class,
                        (JsonSerializer<MenuType>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()))
                .registerTypeAdapter(PredEvalTaskStatus.class,
                        (JsonDeserializer<PredEvalTaskStatus>) (json, type, ctx) ->
                                json.isJsonNull() ? null : PredEvalTaskStatus.fromValue(json.getAsString()))
                .registerTypeAdapter(PredEvalTaskStatus.class,
                        (JsonSerializer<PredEvalTaskStatus>) (src, type, ctx) ->
                                new JsonPrimitive(src.getValue()));
        retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(okHttpClientBuilder.build())
                .addConverterFactory(GsonConverterFactory.create(gsonBuilder.create()))
                .build();

        this.authApiService = retrofit.create(AuthApiService.class);
        this.apiKeyApiService = retrofit.create(ApiKeyApiService.class);
        this.userApiService = retrofit.create(UserApiService.class);
        this.algorithmApiService = retrofit.create(AlgorithmApiService.class);
        this.algorithmSelectApiService = retrofit.create(AlgorithmSelectApiService.class);
        this.favoriteApiService = retrofit.create(FavoriteApiService.class);
        this.recommendationApiService = retrofit.create(RecommendationApiService.class);
        this.datasetApiService = retrofit.create(DatasetApiService.class);
        this.deptApiService = retrofit.create(DeptApiService.class);
        this.dictApiService = retrofit.create(DictApiService.class);
        this.fileApiService = retrofit.create(FileApiService.class);
        this.inputHistoryApiService = retrofit.create(InputHistoryApiService.class);
        this.menuApiService = retrofit.create(MenuApiService.class);
        this.modelApiService = retrofit.create(ModelApiService.class);
        this.roleApiService = retrofit.create(RoleApiService.class);
        this.taskApiService = retrofit.create(TaskApiService.class);
        this.memberApiService = retrofit.create(MemberApiService.class);
        this.packageApiService = retrofit.create(PackageApiService.class);
        this.orderApiService = retrofit.create(OrderApiService.class);
        this.feedbackApiService = retrofit.create(FeedbackApiService.class);
        this.messageApiService = retrofit.create(MessageApiService.class);
        this.announcementApiService = retrofit.create(AnnouncementApiService.class);
        this.messageTemplateApiService = retrofit.create(MessageTemplateApiService.class);
        this.notificationSettingApiService = retrofit.create(NotificationSettingApiService.class);
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
    }
}
