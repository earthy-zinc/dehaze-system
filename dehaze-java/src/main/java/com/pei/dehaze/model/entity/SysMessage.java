package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_message")
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

    private String extra;

    private Integer readStatus;

    private LocalDateTime readTime;

    @TableLogic
    private Integer deleted;

    private LocalDateTime expiresAt;

    @Serial
    private static final long serialVersionUID = 1L;
}
