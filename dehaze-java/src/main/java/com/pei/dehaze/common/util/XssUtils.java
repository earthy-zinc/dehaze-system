package com.pei.dehaze.common.util;

import org.apache.commons.text.StringEscapeUtils;

/**
 * XSS防护工具类
 * 用于过滤和转义用户输入中的HTML特殊字符，防止XSS攻击
 *
 * @author dehaze-system
 * @date 2026-01-10
 */
public class XssUtils {

    /**
     * 清理用户输入，防止XSS攻击
     * 转义HTML特殊字符
     *
     * @param input 用户输入字符串
     * @return 转义后的安全字符串，如果输入为null则返回null
     */
    public static String clean(String input) {
        if (input == null) {
            return null;
        }

        // 使用Apache Commons Text进行HTML转义
        return StringEscapeUtils.escapeHtml4(input);
    }

    /**
     * 清理用户输入，防止XSS攻击
     * 提供默认值，如果输入为null则返回默认值
     *
     * @param input        用户输入字符串
     * @param defaultValue 默认值
     * @return 转义后的安全字符串，如果输入为null则返回默认值
     */
    public static String clean(String input, String defaultValue) {
        String cleaned = clean(input);
        return cleaned != null ? cleaned : defaultValue;
    }

    /**
     * 移除危险的HTML标签
     * 比escapeHtml4更严格的过滤，直接移除script、iframe等标签
     *
     * @param input 用户输入字符串
     * @return 清理后的安全字符串
     */
    public static String stripTags(String input) {
        if (input == null) {
            return null;
        }

        // 移除常见的危险HTML标签
        String cleaned = input.replaceAll("(?i)<script[^>]*>.*?</script>", "")
                .replaceAll("(?i)<iframe[^>]*>.*?</iframe>", "")
                .replaceAll("(?i)<object[^>]*>.*?</object>", "")
                .replaceAll("(?i)<embed[^>]*>.*?</embed>", "")
                .replaceAll("(?i)<link[^>]*>", "")
                .replaceAll("(?i)<meta[^>]*>", "");

        // 再次转义剩余的HTML特殊字符
        return StringEscapeUtils.escapeHtml4(cleaned);
    }
}
