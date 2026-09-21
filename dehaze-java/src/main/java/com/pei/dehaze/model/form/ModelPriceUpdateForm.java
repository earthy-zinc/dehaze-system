package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "模型用户售价版本更新表单（null 视为不修改）")
@Data
public class ModelPriceUpdateForm {

    @Size(max = 24)
    private String unit;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;
}
