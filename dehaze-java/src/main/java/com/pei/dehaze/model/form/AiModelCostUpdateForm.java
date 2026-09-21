package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 成本价格版本更新表单：仅提交需要变更的字段（null 表示不变） */
@Schema(description = "成本单价版本更新表单")
@Data
public class AiModelCostUpdateForm {

    private String currency;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;
}
