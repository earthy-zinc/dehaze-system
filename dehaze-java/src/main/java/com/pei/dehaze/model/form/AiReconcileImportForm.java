package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "供应商账单导入表单")
@Data
public class AiReconcileImportForm {

    @NotBlank(message = "账单内容不能为空")
    private String content;
}
