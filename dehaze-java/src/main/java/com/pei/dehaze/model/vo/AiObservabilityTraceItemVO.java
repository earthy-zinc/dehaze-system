package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 过程链检索行 */
@Schema(description = "过程链检索行")
@Data
public class AiObservabilityTraceItemVO {

    private String traceId;

    private Long conversationId;

    private String conversationTitle;

    private Long messageId;

    private String agentCode;

    private String traceType;

    private String model;

    /** 1:成功;2:失败;3:中断;4:超时 */
    private Integer status;

    private String errorType;

    private Integer durationMs;

    private Integer firstTokenMs;

    private Integer llmCallCount;

    private Integer totalTokens;

    private Integer promptTokens;

    private Integer completionTokens;

    private Integer cachedTokens;

    private Integer stepCount;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
