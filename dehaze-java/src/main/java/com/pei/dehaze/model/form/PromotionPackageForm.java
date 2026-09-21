package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "促销活动关联套餐表单")
public class PromotionPackageForm {

    @Schema(description = "关联套餐ID列表", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotEmpty(message = "关联套餐ID列表不能为空")
    private List<Long> packageIds;
}
