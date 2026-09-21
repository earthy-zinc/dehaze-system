package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.handler.LongListTypeHandler;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.util.List;
import java.util.Map;

/**
 * AI 对话消息
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_message", autoResultMap = true)
public class SysAiMessage extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long conversationId;

    private Long parentMessageId;

    private String role;

    private String content;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> toolCalls;

    private String toolCallId;

    private String model;

    private Integer status;

    private String error;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> metadata;

    private Integer inputTokens;

    private Integer outputTokens;

    private Integer cachedInputTokens;

    private Long credits;

    private String taskId;

    @TableField(typeHandler = LongListTypeHandler.class)
    private List<Long> usedMemoryIds;

    private Integer edited;

    private String originalContent;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
