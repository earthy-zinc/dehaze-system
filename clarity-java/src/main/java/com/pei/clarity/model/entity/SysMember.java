package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

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

    private Integer monthlyDerainQuota;

    private Integer monthlyDerainUsed;

    private Integer monthlyDesnowQuota;

    private Integer monthlyDesnowUsed;

    private Integer monthlyLowlightQuota;

    private Integer monthlyLowlightUsed;

    private Integer monthlySuperResolutionQuota;

    private Integer monthlySuperResolutionUsed;

    private Integer monthlyDenoiseQuota;

    private Integer monthlyDenoiseUsed;

    private Integer monthlyInpaintQuota;

    private Integer monthlyInpaintUsed;

    private Integer monthlyEvaluateQuota;

    private Integer monthlyEvaluateUsed;

    private Integer quotaResetMonth;

    private Integer status;

    private String frozenReason;

    private LocalDateTime frozenTime;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
