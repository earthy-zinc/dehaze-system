package com.pei.clarity.model.vo;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "会员等级权益配置视图对象，字段集与取值口径对齐 python `_benefit_to_vo`")
public class BenefitVO {

    @Schema(description = "等级标识")
    private String levelCode;

    @Schema(description = "等级名称")
    private String levelName;

    @Schema(description = "成长值下限")
    private Long growthMin;

    @Schema(description = "成长值上限（0表示无上限）")
    private Long growthMax;

    @Schema(description = "AI对话日限额（积分/天，每日0点重置）")
    private Long aiCreditsDaily;

    @Schema(description = "AI对话月限额（积分/月，每月1日重置）")
    private Long aiCreditsMonthly;

    @Schema(description = "多模态视觉读取日限额（次/天，每日0点重置）")
    private Integer multimodalLimit;

    @Schema(description = "VIP按月赠送积分（0表示该等级不赠送）")
    private Long vipGiftCredits;

    @Schema(description = "历史记录保留条数")
    private Integer historyRetention;

    @Schema(description = "批量处理上限（张）")
    private Integer batchLimit;

    @Schema(description = "同时在线设备数上限")
    private Integer maxDevices;

    @Schema(description = "处理优先级(1:普通;2:优先;3:高优先;4:最高)")
    private Integer priority;

    @Schema(description = "高级参数调节(0:关闭;1:开启)")
    private Integer advancedParams;

    @Schema(description = "高清图导出(0:关闭;1:开启)")
    private Integer hdExport;

    @Schema(description = "对比报告导出(0:关闭;1:开启)")
    private Integer reportExport;

    @Schema(description = "批量打包下载(0:关闭;1:开启)")
    private Integer batchDownload;

    @Schema(description = "排序值")
    private Integer sort;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description = "月度去雾次数配额")
    private Integer monthlyDehazeQuota;

    @Schema(description = "月度去雨次数配额")
    private Integer monthlyDerainQuota;

    @Schema(description = "月度去雪次数配额")
    private Integer monthlyDesnowQuota;

    @Schema(description = "月度低光增强次数配额")
    private Integer monthlyLowlightQuota;

    @Schema(description = "月度超分辨率次数配额")
    private Integer monthlySuperResolutionQuota;

    @Schema(description = "月度去噪次数配额")
    private Integer monthlyDenoiseQuota;

    @Schema(description = "月度图像修复次数配额")
    private Integer monthlyInpaintQuota;

    @Schema(description = "月度评估次数配额")
    private Integer monthlyEvaluateQuota;
}
