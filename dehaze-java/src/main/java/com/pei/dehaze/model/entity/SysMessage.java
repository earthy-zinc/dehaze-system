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

@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_message", autoResultMap = true)
public class SysMessage extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String type;

    private String title;

    private String content;

    private Integer senderType;

    private Long recipientId;

    private String bizModule;

    private String bizId;

    private Integer priority;

    private String jumpUrl;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> extra;

    private Integer readStatus;

    private LocalDateTime readTime;

    @TableLogic
    private Integer deleted;

    private LocalDateTime expiresAt;

    @Serial
    private static final long serialVersionUID = 1L;
}
