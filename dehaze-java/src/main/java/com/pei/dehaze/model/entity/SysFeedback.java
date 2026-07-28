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
@TableName("sys_feedback")
public class SysFeedback extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String feedbackType;

    private String title;

    private String content;

    private String contact;

    private String images;

    private String relatedModule;

    private Integer status;

    private Integer priority;

    private Long assigneeId;

    private LocalDateTime assignedTime;

    private String tags;

    private String closeReason;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
