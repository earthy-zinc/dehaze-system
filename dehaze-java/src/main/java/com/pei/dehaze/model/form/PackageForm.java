package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.Map;

@Data
@Schema(description = "套餐表单")
public class PackageForm {

    @Schema(description = "套餐ID")
    private Long id;

    @Schema(description = "套餐名称", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "套餐名称不能为空")
    private String name;

    @Schema(description = "会员等级(level_1/level_2/level_3)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "会员等级不能为空")
    private String levelCode;

    @Schema(description = "计费周期(monthly/quarterly/yearly)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "计费周期不能为空")
    private String period;

    @Schema(description = "有效期天数", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "有效期天数不能为空")
    @Min(value = 1, message = "有效期天数必须大于0")
    private Integer periodDays;

    @Schema(description = "原价（分）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "原价不能为空")
    @Min(value = 0, message = "原价不能为负数")
    private Long originalPrice;

    @Schema(description = "促销价（分）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "促销价不能为空")
    @Min(value = 0, message = "促销价不能为负数")
    private Long salePrice;

    @Schema(description = "套餐描述")
    private String description;

    @Schema(description = "权益覆盖项")
    private Map<String, Integer> benefitOverrides;

    @Schema(description = "排序值")
    private Integer sort;

    @Schema(description = "状态(1:上架;0:下架)")
    private Integer status;
}
