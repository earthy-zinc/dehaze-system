package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Schema(description = "优惠券表单")
public class CouponForm {

    @Schema(description = "优惠券ID")
    private Long id;

    @Schema(description = "优惠券名称", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "优惠券名称不能为空")
    private String name;

    @Schema(description = "类型(full_reduction/discount/no_threshold/trial)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "优惠券类型不能为空")
    private String type;

    @Schema(description = "面值", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "面值不能为空")
    @Min(value = 0, message = "面值不能为负数")
    private Long faceValue;

    @Schema(description = "使用门槛（分）")
    private Long threshold;

    @Schema(description = "有效期类型(fixed/relative)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "有效期类型不能为空")
    private String validType;

    @Schema(description = "有效期开始时间")
    private LocalDateTime validStart;

    @Schema(description = "有效期结束时间")
    private LocalDateTime validEnd;

    @Schema(description = "领取后有效天数")
    private Integer validDays;

    @Schema(description = "发放总量(-1为不限)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "发放总量不能为空")
    @Min(value = -1, message = "发放总量不能小于-1")
    private Integer totalQty;

    @Schema(description = "每人限领数量", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "每人限领数量不能为空")
    @Min(value = 1, message = "每人限领数量必须大于0")
    private Integer perUserLimit;

    @Schema(description = "适用套餐ID列表")
    private List<Long> applicableScope;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;
}
