package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 数据项简要视图对象
 * 用于在图片详情中展示所属数据项信息，避免循环引用
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@Schema(description = "数据项简要视图对象")
public class DatasetItemSimpleVO {
    @Schema(description = "数据项ID", example = "1")
    private Long id;

    @Schema(description = "所属数据集ID", example = "1")
    private Long datasetId;

    @Schema(description = "数据项名称，用于标识该数据项", example = "城市街道_001")
    private String name;

    @Schema(description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）", example = "城市街道")
    private String sceneType;

    @Schema(description = "数据项描述信息", example = "城市主干道雾霾场景")
    private String description;
}
