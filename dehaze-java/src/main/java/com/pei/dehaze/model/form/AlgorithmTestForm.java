package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "自定义图片测试表单")
public class AlgorithmTestForm {

    @Schema(description = "文件ID")
    private Long fileId;

    @Schema(description = "图片URL（与fileId二选一）")
    private String imageUrl;

    @Schema(description = "预测参数（JSON）")
    private String params;
}
