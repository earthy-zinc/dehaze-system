package com.pei.clarity.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import lombok.Data;

@Schema(description = "模型用户售价版本更新表单（null 视为不修改）")
@Data
public class ModelPriceUpdateForm {

    @Size(max = 24)
    private String unit;

    @Schema(description = "计费计量单位(null 视为不修改)")
    @Size(max = 32)
    private String billingUnit;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;
}
