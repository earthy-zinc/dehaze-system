package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_member_benefit")
public class SysMemberBenefit extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String levelCode;

    private String levelName;

    private Long growthMin;

    private Long growthMax;

    private Integer monthlyDehazeQuota;

    private Integer monthlyDerainQuota;

    private Integer monthlyDesnowQuota;

    private Integer monthlyLowlightQuota;

    private Integer monthlySuperResolutionQuota;

    private Integer monthlyDenoiseQuota;

    private Integer monthlyInpaintQuota;

    private Integer monthlyEvaluateQuota;

    private Integer historyRetention;

    private Integer batchLimit;

    private Integer maxDevices;

    private Integer priority;

    private Integer advancedParams;

    private Integer hdExport;

    private Integer reportExport;

    private Integer batchDownload;

    private Long aiCreditsDaily;

    private Long aiCreditsMonthly;

    private Integer sort;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
