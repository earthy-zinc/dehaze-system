package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

/**
 * 批量删除请求
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@Schema(description = "批量删除请求")
@Data
public class BatchDeleteRequest {

    @Schema(description = "ID列表", example = "[1, 2, 3]", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotEmpty(message = "ID列表不能为空")
    private List<Long> ids;
}
