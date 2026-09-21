package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
@Schema(description = "套餐表单")
public class PackageForm {

    @Schema(description = "套餐ID")
    private Long id;

    @Schema(description = "套餐名称", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "套餐名称不能为空")
    @Size(min = 2, max = 32, message = "套餐名称长度必须在2-32个字符之间")
    private String name;

    @Schema(description = "商品类型(vip/credit)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "商品类型非法")
    @Pattern(regexp = "vip|credit", message = "商品类型非法")
    private String packageType;

    @Schema(description = "会员等级(level_1/level_2/level_3，会员卡必填)")
    private String levelCode;

    @Schema(description = "计费周期(monthly/quarterly/yearly，会员卡必填)")
    @Pattern(regexp = "monthly|quarterly|yearly", message = "计费周期必须为monthly/quarterly/yearly之一")
    private String period;

    @Schema(description = "有效期天数（会员卡必填）")
    @Min(value = 1, message = "有效期天数必须大于0")
    @Max(value = 365, message = "有效期天数不能超过365")
    private Integer periodDays;

    @Schema(description = "可得积分（积分卡必填，>0）")
    @Min(value = 1, message = "积分卡可得积分必须大于0")
    private Long creditAmount;

    @Schema(description = "原价（分）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "原价不能为空")
    @Min(value = 1, message = "原价必须大于0")
    private Long originalPrice;

    @Schema(description = "促销价（分）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "促销价不能为空")
    @Min(value = 1, message = "促销价必须大于0")
    private Long salePrice;

    @Schema(description = "套餐描述")
    @Size(max = 256, message = "套餐描述长度不能超过256个字符")
    private String description;

    @Schema(description = "权益覆盖项")
    private BenefitOverrides benefitOverrides;

    @Schema(description = "排序值")
    @Min(value = 0, message = "排序值不能为负数")
    @Max(value = 999, message = "排序值不能超过999")
    private Integer sort;

    @Schema(description = "状态(1:上架;0:下架)")
    @Min(value = 0, message = "状态值非法")
    @Max(value = 1, message = "状态值非法")
    private Integer status;
}
