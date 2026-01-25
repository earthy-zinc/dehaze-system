package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 数据项更新表单
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
@Data
@Schema(description = "数据项更新表单")
public class DatasetItemUpdateForm {
    @Schema(
            description = "数据项名称，用于标识该数据项",
            example = "城市街道_001"
    )
    private String name;

    @Schema(
            description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）",
            example = "城市街道"
    )
    private String sceneType;
}
