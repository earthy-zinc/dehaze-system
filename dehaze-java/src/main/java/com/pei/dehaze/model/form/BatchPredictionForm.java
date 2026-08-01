package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Schema(description = "批量预测表单")
@Data
public class BatchPredictionForm {

    @Schema(description = "算法ID")
    @NotNull(message = "算法ID不能为空")
    private Long algorithmId;

    @Schema(description = "批量处理项列表（最多20张）")
    @NotEmpty(message = "处理项不能为空")
    private List<BatchItem> items;

    @Schema(description = "推荐来源标识（可选，来自推荐管理模块）")
    private Long recommendedBy;

    @Schema(description = "批量处理项")
    @Data
    public static class BatchItem {

        @Schema(description = "上传的图片文件ID")
        private Long fileId;

        @Schema(description = "图片URL（与fileId二选一）")
        private String imageUrl;

        @Schema(description = "处理参数（JSON）")
        private String params;
    }
}
