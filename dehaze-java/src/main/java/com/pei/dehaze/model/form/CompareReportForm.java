package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "对比报告生成表单（格式固定为 HTML，无其他可选项）")
@Data
public class CompareReportForm {

    @Schema(description = "处理日志ID（sys_pred_log.id）")
    @NotNull(message = "处理日志ID不能为空")
    private Long logId;
}
