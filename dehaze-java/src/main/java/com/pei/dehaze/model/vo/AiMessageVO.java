package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI 消息响应（含推理步骤）
 *
 * @author dehaze
 */
@Data
public class AiMessageVO {

    private Long id;

    private Long conversationId;

    private Long parentMessageId;

    private String role;

    private String content;

    private List<Map<String, Object>> toolCalls;

    private String toolCallId;

    private String model;

    private Integer status;

    private String error;

    private Map<String, Object> metadata;

    private Integer inputTokens;

    private Integer outputTokens;

    private Integer cachedInputTokens;

    private Long credits;

    private String taskId;

    private Integer edited;

    private String originalContent;

    private List<Long> usedMemoryIds;

    private List<AiAgentThoughtVO> thoughts;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
