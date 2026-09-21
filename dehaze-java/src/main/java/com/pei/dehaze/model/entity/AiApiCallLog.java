package com.pei.dehaze.model.entity;

import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.io.Serial;
import java.io.Serializable;
import java.math.BigDecimal;
import java.time.Instant;

/**
 * AI 兼容 API 调用审计日志（MongoDB 集合 {@code ai_api_call_log}，只追加不更新，TTL 30 天）。
 *
 * <p>写入方是 dehaze-python 兼容协议入口（OpenAI / Claude），文档键为 snake_case，
 * 故字段必须显式 {@link Field} 映射；{@code create_time} 为 UTC 时刻，用 {@link Instant} 承载
 * 以避免 JVM 默认时区参与换算。
 *
 * @author dehaze
 */
@Data
@Document(collection = "ai_api_call_log")
public class AiApiCallLog implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Id
    private String id;

    @Field("user_id")
    private Long userId;

    @Field("key_id")
    private Long keyId;

    /** 脱敏前缀（dhak_xxx...，不存完整 Key） */
    @Field("key_prefix")
    private String keyPrefix;

    @Field("conversation_id")
    private Long conversationId;

    private String model;

    /** chat/completions、messages、models */
    private String endpoint;

    /** openai / claude */
    private String protocol;

    @Field("is_stream")
    private Boolean isStream;

    @Field("input_tokens")
    private Integer inputTokens;

    @Field("output_tokens")
    private Integer outputTokens;

    private BigDecimal credits;

    /** 200/401/403/429/402/5xx */
    @Field("status_code")
    private Integer statusCode;

    @Field("duration_ms")
    private Integer durationMs;

    @Field("client_ip")
    private String clientIp;

    @Field("request_id")
    private String requestId;

    @Field("error_msg")
    private String errorMsg;

    @Field("create_time")
    private Instant createTime;
}
