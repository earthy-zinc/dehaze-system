package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

/**
 * 批量删除图片表单
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@Schema(description = "批量删除图片表单")
public class BatchDeleteForm {

    @Schema(
            description = "图片ID列表，单次最多删除100张",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "[1, 2, 3]"
    )
    @NotEmpty(message = "图片ID列表不能为空")
    private List<Long> ids;
}
