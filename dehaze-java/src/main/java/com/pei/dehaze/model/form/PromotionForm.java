package com.pei.dehaze.model.form;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data
@Schema(description = "促销活动表单")
public class PromotionForm {

    @Schema(description = "活动ID")
    private Long id;

    @Schema(description = "活动名称", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "活动名称不能为空")
    private String name;

    @Schema(description = "活动类型(discount/new_user/holiday/full_reduction)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "活动类型不能为空")
    private String type;

    @Schema(description = "活动描述")
    private String description;

    @Schema(description = "开始时间", requiredMode = Schema.RequiredMode.REQUIRED)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @Schema(description = "结束时间", requiredMode = Schema.RequiredMode.REQUIRED)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime endTime;

    @Schema(description = "活动规则")
    private Map<String, Object> activityRules;

    @Schema(description = "适用套餐ID列表")
    private List<Object> applicableScope;

    @Schema(description = "是否仅新用户(1:是;0:否)")
    @Min(value = 0, message = "newUserOnly取值只能为0或1")
    @Max(value = 1, message = "newUserOnly取值只能为0或1")
    private Integer newUserOnly;

    @Schema(description = "状态(1:启用;0:禁用)")
    @Min(value = 0, message = "status取值只能为0或1")
    @Max(value = 1, message = "status取值只能为0或1")
    private Integer status;
}
