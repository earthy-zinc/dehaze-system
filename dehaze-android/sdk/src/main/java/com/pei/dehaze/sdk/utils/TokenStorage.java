package com.pei.dehaze.sdk.utils;

public interface TokenStorage {
    void saveSessionId(String sessionId);
    String loadSessionId();
    void clearSessionId();
}
