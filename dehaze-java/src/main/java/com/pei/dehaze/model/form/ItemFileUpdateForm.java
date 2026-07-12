package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 数据项图片更新表单（包含标注信息）
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "数据项图片更新表单")
public class ItemFileUpdateForm {

    @Schema(
            description = "图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)",
            example = "hazy"
    )
    private String type;

    @Schema(
            description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）",
            example = "城市街道"
    )
    private String sceneType;

    @Schema(
            description = "雾霾程度，支持多种规范：light/medium/heavy（人工分级）、beta=0.5（β参数）、A=0.8,beta=0.2（A+β双参数），可为空",
            example = "medium"
    )
    private String hazeLevel;

    @Schema(
            description = "图片描述信息",
            example = "城市街道场景的有雾图像"
    )
    private String description;
}
