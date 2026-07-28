package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Schema(description = "会员等级调整表单")
public class MemberLevelAdjustForm {

    @Schema(description = "目标等级(level_0/level_1/level_2/level_3)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "等级不能为空")
    @Pattern(regexp = "^level_[0-3]$", message = "等级必须为level_0/level_1/level_2/level_3")
    private String levelCode;

    @Schema(description = "到期时间（NULL表示成长值维持）")
    private LocalDateTime expireTime;

    @Schema(description = "调整原因", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "调整原因不能为空")
    private String reason;
}
