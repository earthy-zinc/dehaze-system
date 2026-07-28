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
@TableName("sys_member")
public class SysMember extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String levelCode;

    private String levelSource;

    private Long growthValue;

    private Long totalConsumption;

    private LocalDateTime expireTime;

    private LocalDateTime becomeMemberTime;

    private Integer monthlyDehazeQuota;

    private Integer monthlyDehazeUsed;

    private Integer monthlyEvaluateQuota;

    private Integer monthlyEvaluateUsed;

    private Integer quotaResetMonth;

    private Integer status;

    private String frozenReason;

    private LocalDateTime frozenTime;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
