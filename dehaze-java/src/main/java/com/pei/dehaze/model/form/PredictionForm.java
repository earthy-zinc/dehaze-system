package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 模型预测表单
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "模型预测表单")
@Data
public class PredictionForm {

    @Schema(description = "算法ID")
    @NotNull(message = "算法ID不能为空")
    private Long algorithmId;

    @Schema(description = "原始图片文件ID")
    private Long fileId;

    @Schema(description = "原始图片URL（与fileId二选一）")
    private String imageUrl;

    @Schema(description = "预测参数（JSON）")
    private String params;

    @Schema(description = "推荐来源标识（可选，来自推荐管理模块，用于追踪推荐采纳率）")
    private Long recommendedBy;
}
