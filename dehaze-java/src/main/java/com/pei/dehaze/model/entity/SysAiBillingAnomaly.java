package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.time.LocalDateTime;

/** AI 计费异常事件（表 sys_ai_billing_anomaly，只追加） */
@Data
public class SysAiBillingAnomaly {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long billingId;

    /** single_high/burst/consecutive_quota_fail/empty_high_output */
    private String anomalyType;

    private String detail;

    /** 0:待处理;1:已处理;2:已忽略 */
    private Integer status;

    private LocalDateTime triggerAt;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
