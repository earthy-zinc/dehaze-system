package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 效果评估表单
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "效果评估表单")
@Data
public class EvaluationForm {

    @Schema(description = "算法ID")
    @NotNull(message = "算法ID不能为空")
    private Long algorithmId;

    @Schema(description = "预测结果文件ID")
    private Long predFileId;

    @Schema(description = "Ground Truth参考图片文件ID")
    private Long gtFileId;

    @Schema(description = "评估参数（JSON）")
    private String params;
}
