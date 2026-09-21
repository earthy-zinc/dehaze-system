package com.pei.dehaze.common.util;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

/**
 * 计费域日期入参解析。
 *
 * <p>对齐 python {@code ai_billing._parse_datetime}：接受 {@code yyyy-MM-dd HH:mm:ss} 与
 * {@code yyyy-MM-dd}（仅日期按当日 00:00:00），非法格式抛参数错误（A0400）而非 HTTP 400；
 * 额外接受 ISO 8601 便于 SDK 调用。
 */
public final class AiDateTimeUtils {

    private static final DateTimeFormatter DATE_TIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static final DateTimeFormatter DATE = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private AiDateTimeUtils() {
    }

    public static LocalDateTime parse(String value) {
        LocalDateTime parsed = parseOrNull(value);
        if (parsed == null && CharSequenceUtil.isNotBlank(value)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "时间格式不正确: " + value);
        }
        return parsed;
    }

    /**
     * 宽松解析：非法格式返回 null（对齐 python 兼容调用审计「非法按无过滤处理」的口径）
     */
    public static LocalDateTime parseOrNull(String value) {
        if (CharSequenceUtil.isBlank(value)) {
            return null;
        }
        try {
            return LocalDateTime.parse(value, DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        } catch (DateTimeParseException ignored) {
            // 继续尝试空格分隔与仅日期格式
        }
        try {
            return LocalDateTime.parse(value, DATE_TIME);
        } catch (DateTimeParseException ignored) {
            // 继续尝试仅日期格式
        }
        try {
            // 仅日期按当日 00:00:00 处理（python strptime("%Y-%m-%d") 同口径）
            return LocalDate.parse(value, DATE).atStartOfDay();
        } catch (DateTimeParseException ignored) {
            return null;
        }
    }
}
