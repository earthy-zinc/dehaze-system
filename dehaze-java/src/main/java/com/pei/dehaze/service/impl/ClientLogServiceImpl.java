package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.form.ClientLogBatchForm;
import com.pei.dehaze.model.form.ClientLogEntryForm;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.ClientLogService;
import net.logstash.logback.argument.StructuredArguments;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 前端日志接收服务实现。
 * <p>
 * 单条日志 message 最长 2000 字符、error_stack 最长 8000 字符（超长截断）；
 * 匿名（未登录）仅允许上报 ERROR 且必须携带 trace_id；已登录用户从会话解析 user_id 注入。
 * 通过专用 logger（client-log）写入 logs/{yyyy-MM-dd}/client.log（NDJSON）。
 */
@Service
public class ClientLogServiceImpl implements ClientLogService {

    private static final Logger CLIENT_LOGGER = LoggerFactory.getLogger("client-log");

    private static final int MAX_MESSAGE_LENGTH = 2000;
    private static final int MAX_ERROR_STACK_LENGTH = 8000;

    @Override
    public void collect(ClientLogBatchForm form) {
        if (
            form == null 
            || form.getLogs() == null 
            || form.getLogs().isEmpty()
        ) {
            throw new BusinessException("日志列表不能为空");
        }
        for (ClientLogEntryForm entry : form.getLogs()) {
            write(entry);
        }
    }

    private void write(ClientLogEntryForm entry) {
        Long userId = SecurityUtils.getUserId();
        String traceId = normalizeTraceId(entry.getTraceId());
        boolean anonymous = userId == null;

        // 匿名仅允许上报 ERROR 且必须携带 trace_id，否则丢弃该条，避免被滥用刷日志
        if (
            anonymous 
            && (!isError(entry.getLevel()) || CharSequenceUtil.isBlank(traceId))
        ) {
            return;
        }

        String message = truncate(entry.getMessage(), MAX_MESSAGE_LENGTH);
        Map<String, Object> fields = buildFields(entry, userId, traceId);

        // 暂时以本条日志自身的 trace_id / user_id 覆盖请求上下文，保证落盘字段正确
        String originTrace = MDC.get("trace_id");
        String originUser = MDC.get("user_id");
        MDC.put("trace_id", traceId);
        if (userId != null) {
            MDC.put("user_id", userId.toString());
        } else {
            MDC.remove("user_id");
        }

        try {
            logAtLevel(entry.getLevel(), message, fields);
        } finally {
            restoreMdc(originTrace, originUser);
        }
    }

    private void logAtLevel(String level, String message, Map<String, Object> fields) {
        String normalized = CharSequenceUtil.blankToDefault(level, "INFO").toUpperCase();
        switch (normalized) {
            case "ERROR" -> CLIENT_LOGGER.error(message, StructuredArguments.entries(fields));
            case "WARN" -> CLIENT_LOGGER.warn(message, StructuredArguments.entries(fields));
            default -> CLIENT_LOGGER.info(message, StructuredArguments.entries(fields));
        }
    }

    Map<String, Object> buildFields(ClientLogEntryForm entry, Long userId, String traceId) {
        Map<String, Object> fields = new LinkedHashMap<>();
        // 注意：不注入前端 timestamp —— LogstashEncoder 已输出服务端接收时间的 timestamp 字段，
        // 且 logstash 以该字段派生 @timestamp，避免 JSON 同键冲突与时间维度口径不一致。
        putIfNotBlank(fields, "app", entry.getApp());
        putIfNotBlank(fields, "app_version", entry.getAppVersion());
        putIfNotBlank(fields, "url", entry.getUrl());
        putIfNotBlank(fields, "user_agent", entry.getUserAgent());
        putIfNotBlank(fields, "error_type", entry.getErrorType());
        putIfNotBlank(fields, "error_source", entry.getErrorSource());
        putIfNotBlank(fields, "error_stack", truncate(entry.getErrorStack(), MAX_ERROR_STACK_LENGTH));
        putIfNotBlank(fields, "method", entry.getMethod());
        putIfNotBlank(fields, "path", entry.getPath());
        putIfNotBlank(fields, "code", entry.getCode());
        putIfNotBlank(fields, "type", entry.getType());
        putIfNotBlank(fields, "metric_name", entry.getMetricName());
        putIfNotBlank(fields, "navigation_type", entry.getNavigationType());
        putIfNotBlank(fields, "resource_url", entry.getResourceUrl());
        if (entry.getStatus() != null) {
            fields.put("status", entry.getStatus());
        }
        if (entry.getDuration() != null) {
            fields.put("duration", entry.getDuration());
        }
        if (entry.getMetricValue() != null) {
            fields.put("metric_value", entry.getMetricValue());
        }
        // trace_id / user_id 由 MDC 注入，此处避免与 MDC 字段重复
        return fields;
    }

    private static void putIfNotBlank(Map<String, Object> fields, String key, String value) {
        if (CharSequenceUtil.isNotBlank(value)) {
            fields.put(key, value);
        }
    }

    private static boolean isError(String level) {
        return "ERROR".equalsIgnoreCase(level);
    }

    private static String normalizeTraceId(String traceId) {
        return CharSequenceUtil.blankToDefault(traceId, "");
    }

    private static String truncate(String value, int maxLength) {
        if (value == null) {
            return null;
        }
        return value.length() > maxLength ? value.substring(0, maxLength) : value;
    }

    private static void restoreMdc(String traceId, String userId) {
        if (traceId == null) {
            MDC.remove("trace_id");
        } else {
            MDC.put("trace_id", traceId);
        }
        if (userId == null) {
            MDC.remove("user_id");
        } else {
            MDC.put("user_id", userId);
        }
    }
}
