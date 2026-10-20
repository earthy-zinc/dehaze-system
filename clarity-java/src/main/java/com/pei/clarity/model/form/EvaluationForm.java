package com.pei.clarity.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
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

    @Schema(description = "图像模型标识（统一模型规范 slug）")
    @NotBlank(message = "图像模型标识不能为空")
    private String modelId;

    @Schema(description = "预测结果文件ID")
    private Long predFileId;

    @Schema(description = "预测结果图片URL")
    private String predUrl;

    @Schema(description = "Ground Truth参考图片文件ID")
    private Long gtFileId;

    @Schema(description = "Ground Truth参考图片URL")
    private String gtUrl;

    @Schema(description = "评估参数（JSON）")
    private String params;
}
