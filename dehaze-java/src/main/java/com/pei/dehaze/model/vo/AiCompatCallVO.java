package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/** AI 兼容 API 调用审计记录（MongoDB ai_api_call_log） */
@Schema(description = "AI 兼容 API 调用审计记录")
@Data
public class AiCompatCallVO {

    private String id;

    private Long keyId;

    /** 脱敏 Key 前缀 */
    private String keyPrefix;

    private Long conversationId;

    private String model;

    /** chat/completions、messages、models */
    private String endpoint;

    /** openai / claude */
    private String protocol;

    private Boolean isStream;

    private Integer inputTokens;

    private Integer outputTokens;

    private BigDecimal credits;

    /** 200/401/403/429/402/5xx */
    private Integer statusCode;

    private Integer durationMs;

    private String clientIp;

    private String requestId;

    private String errorMsg;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
