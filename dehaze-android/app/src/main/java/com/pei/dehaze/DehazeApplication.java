package com.pei.dehaze;

import android.app.Application;

import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.security.SharedPreferencesTokenStorage;

public class DehazeApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();

        // 1. 先初始化 Token 持久化存储（从磁盘恢复 token）
        TokenManager.init(new SharedPreferencesTokenStorage(this));

        // 2. 再初始化 DehazeSDK（Token 拦截器会读取 TokenManager）
        DehazeSDK.initialize(
            new DehazeSDK.Builder()
                .setBaseUrl("http://10.0.2.2:8989") // Android模拟器访问本机localhost需要使用10.0.2.2
                .setDebug(true)
        );
    }
}