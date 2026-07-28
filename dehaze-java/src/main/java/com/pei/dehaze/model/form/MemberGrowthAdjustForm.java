package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "会员成长值调整表单")
public class MemberGrowthAdjustForm {

    @Schema(description = "变动值（正数增加/负数扣减）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "变动值不能为空")
    private Integer changeValue;

    @Schema(description = "调整原因", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "调整原因不能为空")
    private String reason;
}
