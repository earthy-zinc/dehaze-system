package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 数据项创建表单
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
@Data
@Schema(description = "数据项创建表单")
public class DatasetItemCreateForm {

    @NotNull(message = "数据集ID不能为空")
    @Schema(
            description = "所属数据集ID，指定数据项归属的数据集",
            requiredMode = Schema.RequiredMode.REQUIRED,
            example = "1"
    )
    private Long datasetId;

    @Schema(
            description = "数据项名称，用于标识该数据项（如：scene_001）",
            example = "城市街道_001"
    )
    private String name;

    @Schema(
            description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）",
            example = "outdoor"
    )
    private String sceneType;

    @Schema(
            description = "数据项描述信息",
            example = "城市主干道雾霾场景"
    )
    private String description;
}
