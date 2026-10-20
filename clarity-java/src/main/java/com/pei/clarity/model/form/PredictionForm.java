package com.pei.clarity.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 图像编辑任务提交表单（POST /api/v1/images/edits）
 */
@Schema(description = "图像编辑任务提交表单")
@Data
public class PredictionForm {

    @Schema(description = "图像模型标识（统一模型规范 slug）；为空取平台默认图像模型")
    private String model;

    @Schema(description = "原始图片文件ID")
    private Long fileId;

    @Schema(description = "原始图片URL（与fileId二选一）")
    private String imageUrl;

    @Schema(description = "推理参数（JSON字符串）")
    private String params;

    @Schema(description = "推荐来源标识（可选，来自推荐管理模块，用于追踪推荐采纳率）")
    private Long recommendedBy;
}
