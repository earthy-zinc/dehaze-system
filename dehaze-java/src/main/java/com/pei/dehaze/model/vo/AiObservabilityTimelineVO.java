package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/** 会话审计时间线：轮次（user→assistant 配对）+ 轮内事件按 ts 交织 */
@Schema(description = "会话审计时间线")
@Data
public class AiObservabilityTimelineVO {

    private Conversation conversation;

    private List<Round> rounds = new ArrayList<>();

    @Schema(description = "会话元信息")
    @Data
    public static class Conversation {

        private Long id;

        private String title;

        private Long userId;

        private String agentCode;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }

    @Schema(description = "时间线消息")
    @Data
    public static class Message {

        private Long id;

        private String role;

        private String content;

        private Integer status;

        private String model;

        private Integer inputTokens;

        private Integer outputTokens;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;
    }

    @Schema(description = "轮内事件（kind 区分类型，字段按 kind 取用）")
    @Data
    public static class Event {

        private String kind;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime ts;

        private Message message;

        private Object snapshot;

        private Integer seq;

        private String model;

        private Integer status;

        private Integer durationMs;

        private Integer firstTokenMs;

        private Integer promptTokens;

        private Integer completionTokens;

        private Integer cachedTokens;

        private Object toolCall;

        private Object attempts;

        private Object rawRequest;

        private Object rawResponse;

        private Object summary;

        private Integer position;

        private String tool;

        private String thought;

        private Object toolInput;

        private String observation;

        private Integer latencyMs;

        private String agentCode;

        private Integer isSubagent;

        private String event;

        private Object detail;

        private String billType;

        private Integer credits;

        private Object tokens;
    }

    @Schema(description = "轮内过程链（主对话 + 旁路）")
    @Data
    public static class Trace {

        private String traceId;

        private String traceType;

        private Integer status;

        private String errorType;

        private Object errorDetail;

        private String model;

        private Integer durationMs;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime createTime;

        private List<Event> events = new ArrayList<>();
    }

    @Schema(description = "轮次")
    @Data
    public static class Round {

        private Message userMessage;

        private Message assistantMessage;

        private List<Trace> traces = new ArrayList<>();
    }
}
