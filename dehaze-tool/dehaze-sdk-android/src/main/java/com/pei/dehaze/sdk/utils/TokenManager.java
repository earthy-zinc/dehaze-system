package com.pei.dehaze.sdk.utils;

/**
 * Token管理工具类
 * 用于存储和获取用户认证Token
 */
public class TokenManager {
    private static String token;
    
    /**
     * 设置Token
     *
     * @param token 用户认证Token
     */
    public static void setToken(String token) {
        TokenManager.token = token;
    }
    
    /**
     * 获取Token
     *
     * @return 用户认证Token
     */
    public static String getToken() {
        return token;
    }
    
    /**
     * 清除Token
     */
    public static void clearToken() {
        token = null;
    }
    
    /**
     * 检查Token是否存在
     *
     * @return Token是否存在
     */
    public static boolean hasToken() {
        return token != null && !token.isEmpty();
    }
}