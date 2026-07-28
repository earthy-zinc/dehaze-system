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

    private Integer monthlyEvaluateQuota;

    private Integer historyRetention;

    private Integer batchLimit;

    private Integer priority;

    private Integer advancedParams;

    private Integer hdExport;

    private Integer reportExport;

    private Integer batchDownload;

    private Integer sort;

    private Integer status;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
