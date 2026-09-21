package com.pei.dehaze.model.form;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "权益覆盖项")
public class BenefitOverrides {

    @Schema(description = "月度去雾配额")
    private Integer monthlyDehazeQuota;

    @Schema(description = "月度去雨配额")
    private Integer monthlyDerainQuota;

    @Schema(description = "月度去雪配额")
    private Integer monthlyDesnowQuota;

    @Schema(description = "月度低光增强配额")
    private Integer monthlyLowlightQuota;

    @Schema(description = "月度超分辨率配额")
    private Integer monthlySuperResolutionQuota;

    @Schema(description = "月度去噪配额")
    private Integer monthlyDenoiseQuota;

    @Schema(description = "月度图像修复配额")
    private Integer monthlyInpaintQuota;

    @Schema(description = "月度评价配额")
    private Integer monthlyEvaluateQuota;

    @Schema(description = "AI对话日限额(积分/天)")
    private Long aiCreditsDaily;

    @Schema(description = "AI对话月限额(积分/月)")
    private Long aiCreditsMonthly;

    @Schema(description = "历史保留天数")
    private Integer historyRetention;

    @Schema(description = "批量限制")
    private Integer batchLimit;

    @Schema(description = "优先级")
    private Integer priority;

    @Schema(description = "高级参数")
    private Integer advancedParams;

    @Schema(description = "高清导出")
    private Integer hdExport;

    @Schema(description = "报告导出")
    private Integer reportExport;

    @Schema(description = "批量下载")
    private Integer batchDownload;
}
