package com.pei.dehaze.common.util;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;

/**
 * AI 域跨端 JSON 互认工具。
 *
 * <p>三端共享的 Redis 缓存（ai:model:list / ai:provider:list）与 JSON 列
 * （extra_request_params / user_identity_forward / credentials）由 Python 以
 * snake_case 写入，Java 必须同格式读写才能跨端互认，故单独维护一个
 * snake_case 映射器，与 Web 响应的 camelCase 契约隔离。
 */
public final class AiJsonUtils {

    private static final ObjectMapper SNAKE_MAPPER = JsonMapper.builder()
            .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .addModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .build();

    private AiJsonUtils() {
    }

    public static String write(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return SNAKE_MAPPER.writeValueAsString(value);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "JSON 序列化失败");
        }
    }

    public static <T> T read(String json, TypeReference<T> type) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return SNAKE_MAPPER.readValue(json, type);
        } catch (Exception e) {
            return null;
        }
    }

    public static <T> T read(String json, Class<T> type) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return SNAKE_MAPPER.readValue(json, type);
        } catch (Exception e) {
            return null;
        }
    }
}
