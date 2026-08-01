package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "对比报告生成表单")
@Data
public class CompareReportForm {

    @Schema(description = "处理日志ID（sys_pred_log.id）")
    @NotNull(message = "处理日志ID不能为空")
    private Long logId;

    @Schema(description = "报告格式：pdf 或 image")
    @NotBlank(message = "报告格式不能为空")
    private String format;

    @Schema(description = "是否包含评估指标")
    private Boolean includeMetrics;

    @Schema(description = "是否包含滤镜参数")
    private Boolean includeFilters;
}
