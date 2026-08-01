package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "参数预设表单")
@Data
public class PresetForm {

    @Schema(description = "预设名称")
    @NotBlank(message = "预设名称不能为空")
    private String name;

    @Schema(description = "关联算法ID")
    @NotNull(message = "算法ID不能为空")
    private Long algorithmId;

    @Schema(description = "参数键值对(JSON)")
    @NotBlank(message = "参数不能为空")
    private String params;

    @Schema(description = "是否默认预设")
    private Integer isDefault;
}
