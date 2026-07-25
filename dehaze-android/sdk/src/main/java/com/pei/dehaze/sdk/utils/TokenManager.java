package com.pei.dehaze.sdk.utils;

public class TokenManager {
    private static String sessionId;
    private static TokenStorage storage;
    private static SessionInvalidListener sessionInvalidListener;
    private static boolean isInvalidating = false;

    private TokenManager() {
    }

    public static void init(TokenStorage storage) {
        TokenManager.storage = storage;
        if (storage != null) {
            sessionId = storage.loadSessionId();
        }
    }

    public static void setSessionId(String id) {
        synchronized (TokenManager.class) {
            TokenManager.sessionId = id;
            isInvalidating = false;
            if (storage != null) {
                storage.saveSessionId(id);
            }
        }
    }

    public static String getSessionId() {
        synchronized (TokenManager.class) {
            return sessionId;
        }
    }

    public static void clearAll() {
        synchronized (TokenManager.class) {
            sessionId = null;
            if (storage != null) {
                storage.clearSessionId();
            }
        }
    }

    public static boolean hasToken() {
        synchronized (TokenManager.class) {
            return sessionId != null && !sessionId.isEmpty();
        }
    }

    public static void setSessionInvalidListener(SessionInvalidListener listener) {
        sessionInvalidListener = listener;
    }

    public static void triggerSessionInvalid() {
        synchronized (TokenManager.class) {
            if (isInvalidating) {
                return;
            }
            isInvalidating = true;
            sessionId = null;
            if (storage != null) {
                storage.clearSessionId();
            }
        }
        if (sessionInvalidListener != null) {
            sessionInvalidListener.onSessionInvalid();
        }
    }

    private static final String CODE_TOKEN_INVALID = "A0230";
    private static final String CODE_TOKEN_FORBIDDEN = "A0231";

    public static boolean isTokenInvalidCode(String code) {
        return CODE_TOKEN_INVALID.equals(code) || CODE_TOKEN_FORBIDDEN.equals(code);
    }
}
