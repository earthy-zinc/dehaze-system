package com.pei.clarity.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "参数预设表单")
@Data
public class PresetForm {

    @Schema(description = "预设名称")
    @NotBlank(message = "预设名称不能为空")
    private String name;

    @Schema(description = "关联图像模型标识（统一模型规范 slug）")
    @NotBlank(message = "图像模型标识不能为空")
    private String modelId;

    @Schema(description = "参数键值对(JSON)")
    @NotBlank(message = "参数不能为空")
    private String params;

    @Schema(description = "是否默认预设")
    private Integer isDefault;
}
