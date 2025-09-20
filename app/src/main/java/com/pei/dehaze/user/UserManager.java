package com.pei.dehaze.user;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.pei.dehaze.model.response.LoginResponse;
import com.pei.dehaze.common.Result;

import lombok.Getter;

/**
 * 用户管理类，负责管理用户登录状态、Token存储和获取
 */
public class UserManager {
    private static final String TAG = "UserManager";
    private static final String PREF_NAME = "user_prefs";
    private static final String KEY_TOKEN = "auth_token";
    private static final String KEY_USERNAME = "username";
    private static final String KEY_USER_ID = "user_id";
    private static final String KEY_IS_LOGGED_IN = "is_logged_in";

    private static UserManager instance;
    private SharedPreferences preferences;
    @Getter
    private String token;
    @Getter
    private String username;
    @Getter
    private String userId;
    @Getter
    private boolean isLoggedIn;

    private UserManager() {
        // 私有构造函数，单例模式
    }

    /**
     * 初始化UserManager
     */
    public static synchronized void init(Context context) {
        if (instance == null) {
            instance = new UserManager();
            instance.preferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            instance.loadUserInfo();
        }
    }

    /**
     * 获取UserManager实例
     */
    public static synchronized UserManager getInstance() {
        if (instance == null) {
            throw new IllegalStateException("UserManager not initialized. Call init() first.");
        }
        return instance;
    }

    /**
     * 从SharedPreferences加载用户信息
     */
    private void loadUserInfo() {
        token = preferences.getString(KEY_TOKEN, null);
        username = preferences.getString(KEY_USERNAME, null);
        userId = preferences.getString(KEY_USER_ID, null);
        isLoggedIn = preferences.getBoolean(KEY_IS_LOGGED_IN, false);
        Log.d(TAG, "Loaded user info: username=" + username + ", isLoggedIn=" + isLoggedIn);
    }

    /**
     * 保存用户登录信息
     */
    public void saveLoginInfo(Result<LoginResponse> res) {
        if (res != null && res.getData() != null) {
            token = res.getData().getToken();

            if (res.getData().getUserInfo() != null) {
                username = res.getData().getUserInfo().getUsername();
                userId = res.getData().getUserInfo().getId();
            }

            isLoggedIn = true;

            // 保存到SharedPreferences
            SharedPreferences.Editor editor = preferences.edit();
            editor.putString(KEY_TOKEN, token);
            editor.putString(KEY_USERNAME, username);
            editor.putString(KEY_USER_ID, userId);
            editor.putBoolean(KEY_IS_LOGGED_IN, true);
            editor.apply();

            Log.d(TAG, "User login info saved: username=" + username);
        }
    }

    /**
     * 清除用户登录信息
     */
    public void clearUserInfo() {
        token = null;
        username = null;
        userId = null;
        isLoggedIn = false;

        // 清除SharedPreferences中的数据
        SharedPreferences.Editor editor = preferences.edit();
        editor.remove(KEY_TOKEN);
        editor.remove(KEY_USERNAME);
        editor.remove(KEY_USER_ID);
        editor.putBoolean(KEY_IS_LOGGED_IN, false);
        editor.apply();

        Log.d(TAG, "User login info cleared");
    }
}