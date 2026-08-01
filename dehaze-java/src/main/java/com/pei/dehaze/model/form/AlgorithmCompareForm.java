package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "算法对比表单")
public class AlgorithmCompareForm {

    @Schema(description = "算法ID列表（2-3个）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "算法ID列表不能为空")
    @Size(min = 2, max = 3, message = "对比算法数量需在2-3个之间")
    private List<Long> algorithmIds;

    @Schema(description = "文件ID")
    private Long fileId;

    @Schema(description = "图片URL（与fileId二选一）")
    private String imageUrl;
}
