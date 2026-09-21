package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 会话响应（含管理端审计视角附加字段）
 *
 * @author dehaze
 */
@Data
public class AiConversationVO {

    private Long id;

    private Long userId;

    private String userName;

    private Long tokenConsumed;

    private Long creditsConsumed;

    private String anomalyType;

    private String anomalyLabel;

    private String title;

    private String model;

    private String agentCode;

    private Integer agentVersion;

    private String summary;

    private String systemPrompt;

    private Map<String, Object> modelConfig;

    private Integer suggestionsEnabled;

    private Long apiKeyId;

    private Integer messageCount;

    private LocalDateTime lastMessageAt;

    private Long currentBranchMessageId;

    private Long lastReadMessageId;

    private Integer pinned;

    private LocalDateTime pinnedAt;

    private LocalDateTime deleteTime;

    private Integer unreadCount;

    private String titleSource;

    private Integer status;

    private Long matchedMessageId;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
