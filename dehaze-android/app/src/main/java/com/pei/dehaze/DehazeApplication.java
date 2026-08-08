package com.pei.dehaze;

import android.app.Activity;
import android.app.Application;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.logger.ConsoleTransport;
import com.pei.dehaze.sdk.logger.FileTransport;
import com.pei.dehaze.sdk.logger.LogEntry;
import com.pei.dehaze.sdk.logger.LogLevel;
import com.pei.dehaze.sdk.logger.LogTransport;
import com.pei.dehaze.sdk.logger.Logger;
import com.pei.dehaze.sdk.logger.RemoteTransport;
import com.pei.dehaze.sdk.utils.TokenManager;

import com.pei.dehaze.security.SharedPreferencesTokenStorage;

import okhttp3.OkHttpClient;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DehazeApplication extends Application {

    private static volatile Activity currentActivity;

    @Override
    public void onCreate() {
        super.onCreate();

        // 初始化 Logger（崩溃捕获依赖，须在 DehazeSDK 之前完成）
        initLogger();

        // 注册全局未捕获异常处理器（error_type: native）
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            Logger.getInstance().error("Uncaught native exception", new LogEntry(
                    LogLevel.ERROR, "", "", "")
                    .setErrorType("native")
                    .setErrorSource("UncaughtExceptionHandler")
                    .setErrorStack(android.util.Log.getStackTraceString(throwable)));
        });

        TokenManager.init(new SharedPreferencesTokenStorage(this));

        DehazeSDK.initialize(
            new DehazeSDK.Builder()
                .setBaseUrl(BuildConfig.BASE_URL)
                .setDebug(BuildConfig.DEBUG)
        );

        registerActivityLifecycleCallbacks(new ActivityLifecycleCallbacks() {
            @Override
            public void onActivityCreated(@NonNull Activity activity, @Nullable Bundle savedInstanceState) {
            }

            @Override
            public void onActivityStarted(@NonNull Activity activity) {
                currentActivity = activity;
            }

            @Override
            public void onActivityResumed(@NonNull Activity activity) {
                currentActivity = activity;
            }

            @Override
            public void onActivityPaused(@NonNull Activity activity) {
            }

            @Override
            public void onActivityStopped(@NonNull Activity activity) {
                if (currentActivity == activity) {
                    currentActivity = null;
                }
            }

            @Override
            public void onActivitySaveInstanceState(@NonNull Activity activity, @NonNull Bundle outState) {
            }

            @Override
            public void onActivityDestroyed(@NonNull Activity activity) {
                if (currentActivity == activity) {
                    currentActivity = null;
                }
            }
        });

        TokenManager.setSessionInvalidListener(() -> {
            Activity activity = currentActivity;
            if (activity == null) {
                return;
            }
            activity.runOnUiThread(() -> {
                if (activity instanceof SessionInvalidHandler) {
                    ((SessionInvalidHandler) activity).onSessionInvalid();
                }
            });
        });
    }

    /**
     * 初始化 Logger（多 transport 架构 §3.6）：
     * - ConsoleTransport（始终开启，Logcat）
     * - FileTransport（开发 7 天 / 生产 3 天兜底，写 filesDir/logs/）
     * - RemoteTransport（生产上报）
     */
    private void initLogger() {
        File filesDir = getFilesDir();
        List<LogTransport> transports =
                new ArrayList<>();
        transports.add(new ConsoleTransport());
        transports.add(new FileTransport(filesDir, BuildConfig.DEBUG ? 7 : 3));
        // 生产环境添加远程上报（崩溃兜底：FileTransport 保留 3 天）
        if (!BuildConfig.DEBUG) {
            OkHttpClient client = new OkHttpClient();
            String baseUrl = BuildConfig.BASE_URL;
            if (baseUrl.endsWith("/")) {
                baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
            }
            transports.add(new RemoteTransport(client, baseUrl));
        }

        Logger.init(
                "android",
                BuildConfig.VERSION_NAME,
                transports
        );

        // 生产环境启动补报（崩溃后本地文件补报）
        if (!BuildConfig.DEBUG) {
            Logger.getInstance().flushFromDisk();
        }
    }

    public static Activity getCurrentActivity() {
        return currentActivity;
    }

    public interface SessionInvalidHandler {
        void onSessionInvalid();
    }
}
