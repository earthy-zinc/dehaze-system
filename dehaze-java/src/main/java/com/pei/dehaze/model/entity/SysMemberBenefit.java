package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import java.io.Serial;
import lombok.Data;
import lombok.EqualsAndHashCode;

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

    /** VIP 按月赠送积分（0 表示该等级不赠送），由 grantVipMonthlyGift 任务发放 */
    private Long vipGiftCredits;

    /** 多模态视觉读取日限额（次/天，按用户全局日计数，每日 0 点重置） */
    private Integer multimodalLimit;

    private Integer sort;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
