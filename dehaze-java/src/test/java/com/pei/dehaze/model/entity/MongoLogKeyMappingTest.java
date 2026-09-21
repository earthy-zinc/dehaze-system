package com.pei.dehaze.model.entity;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.lang.reflect.Modifier;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Mongo 审计日志实体键名守卫。
 *
 * <p>{@code login_log} / {@code audit_log} / {@code ai_api_call_log} 由 java 与 dehaze-python
 * 共写共读，python 按 snake_case 写键；实体一旦漏掉 {@code @Field} 映射（如 userId 直落 userId），
 * 同集合内两端数据就互不可见且不会报错，故此处按「Java 字段名 → 实际落库键名」逐项断言。
 */
@DisplayName("Mongo 审计日志实体键名守卫")
class MongoLogKeyMappingTest {

    @Test
    @DisplayName("login_log 实体键名与 python 写入的 snake_case 一致")
    void loginLogFieldsUseSnakeCaseKeys() {
        assertThat(collection(LoginLog.class)).isEqualTo("login_log");
        Map<String, String> keys = persistedKeys(LoginLog.class);
        assertThat(keys).containsEntry("userId", "user_id")
                .containsEntry("deviceType", "device_type")
                .containsEntry("createTime", "create_time")
                .containsEntry("username", "username")
                .containsEntry("status", "status");
    }

    @Test
    @DisplayName("audit_log 实体键名与 python 写入的 snake_case 一致")
    void auditLogFieldsUseSnakeCaseKeys() {
        assertThat(collection(AuditLog.class)).isEqualTo("audit_log");
        Map<String, String> keys = persistedKeys(AuditLog.class);
        assertThat(keys).containsEntry("operatorId", "operator_id")
                .containsEntry("targetType", "target_type")
                .containsEntry("targetId", "target_id")
                .containsEntry("beforeValue", "before_value")
                .containsEntry("afterValue", "after_value")
                .containsEntry("userAgent", "user_agent")
                .containsEntry("createTime", "create_time");
    }

    @Test
    @DisplayName("ai_api_call_log 实体键名与 python 写入的 snake_case 一致")
    void aiApiCallLogFieldsUseSnakeCaseKeys() {
        assertThat(collection(AiApiCallLog.class)).isEqualTo("ai_api_call_log");
        Map<String, String> keys = persistedKeys(AiApiCallLog.class);
        assertThat(keys).containsEntry("userId", "user_id")
                .containsEntry("keyPrefix", "key_prefix")
                .containsEntry("isStream", "is_stream")
                .containsEntry("statusCode", "status_code")
                .containsEntry("createTime", "create_time");
    }

    private static String collection(Class<?> type) {
        Document document = type.getAnnotation(Document.class);
        return document == null ? null : document.collection();
    }

    /** Java 字段名 → 落库键名（无 @Field 时即字段名本身） */
    private static Map<String, String> persistedKeys(Class<?> type) {
        Map<String, String> keys = new LinkedHashMap<>();
        for (java.lang.reflect.Field field : type.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            Field mapped = field.getAnnotation(Field.class);
            keys.put(field.getName(), mapped == null ? field.getName() : mapped.value());
        }
        return keys;
    }
}
