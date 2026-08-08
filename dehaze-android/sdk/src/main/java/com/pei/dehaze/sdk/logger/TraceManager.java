package com.pei.dehaze.sdk.logger;

import java.security.SecureRandom;

/**
 * trace_id 生成与当前请求 trace 管理。
 *
 * 与后端 §4.3 约定一致：uuid hex 32 位无连字符，透传请求头 X-Trace-Id。
 */
public class TraceManager {

    private static final char[] HEX = "0123456789abcdef".toCharArray();
    private static final SecureRandom RANDOM = new SecureRandom();
    private static String currentTraceId = "";

    private TraceManager() {
    }

    /** 生成 32 位 hex 无连字符 trace_id。 */
    public static String generateTraceId() {
        byte[] bytes = new byte[16];
        RANDOM.nextBytes(bytes);
        char[] chars = new char[32];
        for (int i = 0; i < bytes.length; i++) {
            chars[i * 2] = HEX[(bytes[i] & 0xF0) >>> 4];
            chars[i * 2 + 1] = HEX[bytes[i] & 0x0F];
        }
        return new String(chars);
    }

    public static String getCurrentTraceId() {
        return currentTraceId;
    }

    public static void setCurrentTraceId(String traceId) {
        currentTraceId = traceId;
    }

    /** 在请求入口复用/生成 trace_id 并注入上下文。 */
    public static String ensureTraceId() {
        if (currentTraceId == null || currentTraceId.isEmpty()) {
            currentTraceId = generateTraceId();
        }
        return currentTraceId;
    }

    /** 响应头 X-Trace-Id 回写对齐。 */
    public static void alignTraceId(String traceId) {
        if (traceId != null && !traceId.isEmpty()) {
            currentTraceId = traceId;
        }
    }
}
