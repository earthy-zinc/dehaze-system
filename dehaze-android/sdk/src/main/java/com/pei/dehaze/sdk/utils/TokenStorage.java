package com.pei.dehaze.sdk.utils;

/**
 * Token 持久化存储接口（依赖反转）
 * App 层提供实现（如基于 EncryptedSharedPreferences），SDK 通过此接口持久化 Token
 */
public interface TokenStorage {
    /** 保存 accessToken */
    void saveToken(String token);
    /** 读取 accessToken，不存在返回 null */
    String loadToken();
    /** 清除 accessToken */
    void clearToken();
    /** 保存 refreshToken */
    void saveRefreshToken(String refreshToken);
    /** 读取 refreshToken，不存在返回 null */
    String loadRefreshToken();
    /** 清除 refreshToken */
    void clearRefreshToken();
}
