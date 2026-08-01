package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "图像分析请求表单")
public class AnalyzeForm {

    @Schema(description = "已上传图片ID（与imageUrl二选一）")
    private Long imageId;

    @Schema(description = "图片URL（与imageId二选一）")
    private String imageUrl;
}
