package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "菜单显示状态表单")
@Data
public class MenuVisibleBody {

    @Schema(description = "显示状态(1:显示;0:隐藏)")
    @NotNull(message = "显示状态不能为空")
    @Min(value = 0, message = "显示状态不合法")
    @Max(value = 1, message = "显示状态不合法")
    private Integer visible;
}
