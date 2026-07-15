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
        return StringEscapeUtils.escapeHtml4(input);
    }
}
