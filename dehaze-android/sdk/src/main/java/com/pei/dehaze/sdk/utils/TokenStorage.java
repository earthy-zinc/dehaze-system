package com.pei.dehaze.sdk.utils;

/**
 * Token 持久化存储接口（依赖反转）
 * App 层提供实现（如基于 EncryptedSharedPreferences），SDK 通过此接口持久化 Token
 */
public interface TokenStorage {
    /** 保存 token */
    void saveToken(String token);
    /** 读取 token，不存在返回 null */
    String loadToken();
    /** 清除 token */
    void clearToken();
}
