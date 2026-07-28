package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "优惠券批量发放表单")
public class CouponBatchDistributeForm {

    @Schema(description = "优惠券ID", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "优惠券ID不能为空")
    private Long couponId;

    @Schema(description = "发放范围(all/level/users)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "发放范围不能为空")
    private String targetScope;

    @Schema(description = "等级编码列表（targetScope=level时使用）")
    private List<String> levelCodes;

    @Schema(description = "用户ID列表（targetScope=users时使用）")
    private List<Long> userIds;
}
