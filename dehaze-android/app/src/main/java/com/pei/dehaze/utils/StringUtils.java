package com.pei.dehaze.utils;

import android.widget.EditText;

/**
 * 字符串工具类，消除项目中重复的 safe/getText/safeParseInt 等方法
 */
public final class StringUtils {
    private StringUtils() {}

    /** null 安全的字符串显示：null 返回空串 */
    public static String safe(String s) {
        return s == null ? "" : s;
    }

    /** null 安全的字符串显示：null 返回 fallback */
    public static String safe(String s, String fallback) {
        return s == null ? fallback : s;
    }

    /** 安全 parseInt，失败返回默认值 */
    public static int safeParseInt(String value, int defaultValue) {
        if (value == null) return defaultValue;
        try { return Integer.parseInt(value); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    /** 安全 parseLong，失败返回默认值 */
    public static long safeParseLong(String value, long defaultValue) {
        if (value == null) return defaultValue;
        try { return Long.parseLong(value); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    /** 从 EditText 获取 trim 后的文本，null 安全 */
    public static String getText(EditText et) {
        return et != null && et.getText() != null ? et.getText().toString().trim() : "";
    }
}
