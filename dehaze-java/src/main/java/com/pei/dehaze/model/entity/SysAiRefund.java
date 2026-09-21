package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 积分误扣退款申请（表 sys_ai_refund，状态机 1待审核→2已通过/3已驳回） */
@Data
public class SysAiRefund {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long billingId;

    private Integer amount;

    private String reason;

    private Integer status;

    private Long auditorId;

    private String auditRemark;

    /** 申请人 ID（用户申请退款时记录） */
    @TableField(fill = FieldFill.INSERT)
    private Long createBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}
