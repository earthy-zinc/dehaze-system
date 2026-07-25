package com.pei.dehaze;

import android.app.Activity;
import android.app.Application;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.security.SharedPreferencesTokenStorage;

public class DehazeApplication extends Application {

    private static volatile Activity currentActivity;

    @Override
    public void onCreate() {
        super.onCreate();

        TokenManager.init(new SharedPreferencesTokenStorage(this));

        DehazeSDK.initialize(
            new DehazeSDK.Builder()
                .setBaseUrl(BuildConfig.BASE_URL)
                .setDebug(true)
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

    public static Activity getCurrentActivity() {
        return currentActivity;
    }

    public interface SessionInvalidHandler {
        void onSessionInvalid();
    }
}
