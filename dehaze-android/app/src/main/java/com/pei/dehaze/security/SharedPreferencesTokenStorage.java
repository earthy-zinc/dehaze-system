package com.pei.dehaze.security;

import android.content.Context;
import android.content.SharedPreferences;

import com.pei.dehaze.sdk.utils.TokenStorage;

/**
 * 基于 SharedPreferences 的 Token 持久化实现
 * SDK 通过 TokenStorage 接口完成依赖反转，App 层提供具体存储实现
 */
public class SharedPreferencesTokenStorage implements TokenStorage {

    private static final String PREF_NAME = "dehaze_auth_prefs";
    private static final String KEY_TOKEN = "access_token";
    private static final String KEY_REFRESH_TOKEN = "refresh_token";

    private final SharedPreferences prefs;

    public SharedPreferencesTokenStorage(Context context) {
        // 使用 ApplicationContext 避免内存泄漏
        this.prefs = context.getApplicationContext()
                .getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    @Override
    public void saveToken(String token) {
        prefs.edit().putString(KEY_TOKEN, token).apply();
    }

    @Override
    public String loadToken() {
        return prefs.getString(KEY_TOKEN, null);
    }

    @Override
    public void clearToken() {
        prefs.edit().remove(KEY_TOKEN).apply();
    }

    @Override
    public void saveRefreshToken(String refreshToken) {
        prefs.edit().putString(KEY_REFRESH_TOKEN, refreshToken).apply();
    }

    @Override
    public String loadRefreshToken() {
        return prefs.getString(KEY_REFRESH_TOKEN, null);
    }

    @Override
    public void clearRefreshToken() {
        prefs.edit().remove(KEY_REFRESH_TOKEN).apply();
    }
}
