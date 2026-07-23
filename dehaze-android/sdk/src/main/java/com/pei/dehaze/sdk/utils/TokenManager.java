package com.pei.dehaze.sdk.utils;

/**
 * Token 管理工具类
 * 支持持久化存储（通过 TokenStorage 接口）和 401 自动清理
 */
public class TokenManager {
    private static String token;
    private static String refreshToken;
    private static TokenStorage storage;

    /** Token 无效的业务错误码（A0230/A0231） */
    private static final String CODE_TOKEN_INVALID = "A0230";
    private static final String CODE_TOKEN_FORBIDDEN = "A0231";

    private TokenManager() {
    }

    /**
     * 初始化 Token 持久化存储
     *
     * @param storage 持久化存储实现（App 层传入，如 EncryptedSharedPreferences）
     */
    public static void init(TokenStorage storage) {
        TokenManager.storage = storage;
        // 初始化时从持久层恢复 token
        if (storage != null) {
            token = storage.loadToken();
            refreshToken = storage.loadRefreshToken();
        }
    }

    /**
     * 设置 accessToken（同时持久化）
     *
     * @param token 用户认证 accessToken
     */
    public static void setToken(String token) {
        synchronized (TokenManager.class) {
            TokenManager.token = token;
            if (storage != null) {
                storage.saveToken(token);
            }
        }
    }

    /**
     * 获取 accessToken
     *
     * @return 用户认证 accessToken，不存在返回 null
     */
    public static String getToken() {
        synchronized (TokenManager.class) {
            return token;
        }
    }

    /**
     * 设置 refreshToken（同时持久化）
     *
     * @param refreshToken 刷新令牌
     */
    public static void setRefreshToken(String refreshToken) {
        synchronized (TokenManager.class) {
            TokenManager.refreshToken = refreshToken;
            if (storage != null) {
                storage.saveRefreshToken(refreshToken);
            }
        }
    }

    /**
     * 获取 refreshToken
     *
     * @return refreshToken，不存在返回 null
     */
    public static String getRefreshToken() {
        synchronized (TokenManager.class) {
            return refreshToken;
        }
    }

    /**
     * 清除 accessToken（同时清除持久化）
     */
    public static void clearToken() {
        synchronized (TokenManager.class) {
            token = null;
            if (storage != null) {
                storage.clearToken();
            }
        }
    }

    /**
     * 清除 refreshToken（同时清除持久化）
     */
    public static void clearRefreshToken() {
        synchronized (TokenManager.class) {
            refreshToken = null;
            if (storage != null) {
                storage.clearRefreshToken();
            }
        }
    }

    /**
     * 清除全部 Token（accessToken + refreshToken），用于注销或刷新失败
     */
    public static void clearAll() {
        synchronized (TokenManager.class) {
            token = null;
            refreshToken = null;
            if (storage != null) {
                storage.clearToken();
                storage.clearRefreshToken();
            }
        }
    }

    /**
     * 检查 Token 是否存在
     *
     * @return Token 是否存在
     */
    public static boolean hasToken() {
        synchronized (TokenManager.class) {
            return token != null && !token.isEmpty();
        }
    }

    /**
     * 判断业务错误码是否表示 Token 无效（需要重新登录）
     *
     * @param code 业务错误码
     * @return 是否 Token 无效
     */
    public static boolean isTokenInvalidCode(String code) {
        return CODE_TOKEN_INVALID.equals(code) || CODE_TOKEN_FORBIDDEN.equals(code);
    }
}
