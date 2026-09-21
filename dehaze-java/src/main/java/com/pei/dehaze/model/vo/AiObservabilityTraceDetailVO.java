package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/** 过程链详情：trace 汇总字段 + 上下文快照 + LLM 调用回放 + 推理步骤 + 会话消息 + 计费与产物 */
@Schema(description = "过程链详情")
@Data
@EqualsAndHashCode(callSuper = true)
public class AiObservabilityTraceDetailVO extends AiObservabilityTraceItemVO {

    /** 上下文构成快照 JSON */
    private Object contextSnapshot;

    private Object errorDetail;

    private List<LlmCall> llmCalls = new ArrayList<>();

    private List<AiAgentThoughtVO> thoughts = new ArrayList<>();

    private List<Message> messages = new ArrayList<>();

    private List<Billing> billing = new ArrayList<>();

    private List<Artifact> artifacts = new ArrayList<>();

    @Schema(description = "LLM 调用明细（按 seq 正序）")
    @Data
    public static class LlmCall {

        private Integer seq;

        private Integer stepPosition;

        private String model;

        private Integer status;

        private String errorType;

        private Integer durationMs;

        private Integer firstTokenMs;

        private Integer promptTokens;

        private Integer completionTokens;

        private Integer cachedTokens;

        private Object toolCall;

        private Object inputSnapshot;

        private Object outputSnapshot;

        private Object attempts;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime startTime;

        private Object rawRequest;

        private Object rawResponse;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }

    @Schema(description = "会话消息")
    @Data
    public static class Message {

        private Long id;

        private Long conversationId;

        private Long parentMessageId;

        private String role;

        private String content;

        private Integer status;

        private String model;

        private Integer inputTokens;

        private Integer outputTokens;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }

    @Schema(description = "关联计费记录")
    @Data
    public static class Billing {

        private String billType;

        private String model;

        private String actualModel;

        private Long providerId;

        private Integer inputTokens;

        private Integer outputTokens;

        private Integer cachedInputTokens;

        private Integer credits;

        private Integer creditsSaved;

        private String errorCode;

        private Integer latencyMs;

        private String requestId;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }

    @Schema(description = "关联中间产物")
    @Data
    public static class Artifact {

        private Long id;

        private String type;

        private Object summary;

        private String refType;

        private Long refId;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }
}
