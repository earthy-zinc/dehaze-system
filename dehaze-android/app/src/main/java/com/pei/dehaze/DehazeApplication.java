package com.pei.dehaze;

import android.app.Application;

import com.pei.dehaze.sdk.DehazeSDK;

public class DehazeApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        
        // 初始化DehazeSDK
        DehazeSDK.initialize(
            new DehazeSDK.Builder()
                .setBaseUrl("http://10.0.2.2:8989") // Android模拟器访问本机localhost需要使用10.0.2.2
                .setDebug(true)
        );
    }
}