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

    @Schema(description = "月度评价配额")
    private Integer monthlyEvaluateQuota;

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
