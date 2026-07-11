package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 算法审核表单
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "算法审核表单")
@Data
public class AlgorithmAuditForm {

    @Schema(description = "审核结果：true=通过, false=驳回")
    @NotNull(message = "审核结果不能为空")
    private Boolean approved;

    @Schema(description = "审核备注，驳回时必填")
    private String remark;
}
