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
@TableName("sys_announcement")
public class SysAnnouncement extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String title;

    private String content;

    private String type;

    private Integer importance;

    private String targetScope;

    private String targetParams;

    private Integer status;

    private LocalDateTime sendTime;

    private LocalDateTime expireTime;

    private Integer sentCount;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
