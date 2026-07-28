package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_feedback_reply")
public class SysFeedbackReply extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long feedbackId;

    private Long replierId;

    private Integer replierType;

    private String content;

    private String replyType;

    private String attachments;

    @Serial
    private static final long serialVersionUID = 1L;
}
