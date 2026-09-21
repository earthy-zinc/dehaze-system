package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 对话会话
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_conversation", autoResultMap = true)
public class SysAiConversation extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String title;

    private String model;

    private String agentCode;

    private Integer agentVersion;

    private String summary;

    private Long summaryUptoMessageId;

    private String systemPrompt;

    @TableField(typeHandler = JacksonTypeHandler.class)
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

    private String titleSource;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
