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
import java.util.List;

/**
 * AI 消息反馈
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_message_feedback", autoResultMap = true)
public class SysAiMessageFeedback extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long messageId;

    private Long userId;

    private Long conversationId;

    private String model;

    @TableField("`source`")
    private String source;

    private Integer rating;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<String> tags;

    @TableField("`comment`")
    private String comment;

    private Integer processed;

    private LocalDateTime processTime;

    @TableLogic(value = "0", delval = "1")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
