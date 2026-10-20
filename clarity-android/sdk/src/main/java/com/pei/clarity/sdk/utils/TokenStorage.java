package com.pei.clarity.sdk.utils;

public interface TokenStorage {
    void saveSessionId(String sessionId);
    String loadSessionId();
    void clearSessionId();
}
