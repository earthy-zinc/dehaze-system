package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "自动续费配置表单")
public class AutoRenewConfigForm {

    @Schema(description = "套餐ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "套餐ID不能为空")
    private Long packageId;

    @Schema(description = "支付方式", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "支付方式不能为空")
    private String payMethod;

    @Schema(description = "是否启用", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "启用状态不能为空")
    private Boolean enabled;
}
