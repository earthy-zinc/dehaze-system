package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
@Schema(description = "会员状态变更表单")
public class MemberStatusForm {

    @Schema(description = "状态(1:正常;0:冻结)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "状态不能为空")
    @Min(value = 0, message = "状态值必须为0或1")
    @Max(value = 1, message = "状态值必须为0或1")
    private Integer status;

    @Schema(description = "冻结原因（冻结时必填）")
    private String reason;
}
