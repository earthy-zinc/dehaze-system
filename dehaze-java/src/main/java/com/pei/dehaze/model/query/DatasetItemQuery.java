package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图片列表查询参数
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "数据项查询参数")
public class DatasetItemQuery extends BasePageQuery {

    @Schema(
            description = "数据集ID，用于筛选指定数据集下的数据项",
            example = "1"
    )
    private Long datasetId;

    @Schema(
            description = "搜索关键词，支持按文件名、描述、场景类型模糊搜索",
            example = "城市街道"
    )
    private String keyword;

    @Schema(
            description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）",
            example = "城市街道"
    )
    private String sceneType;

    @Schema(
            description = "雾霾程度，支持多种规范的字符串精确匹配：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等",
            example = "medium"
    )
    private String hazeLevel;

    @Schema(
            description = "最小图片宽度（像素），用于分辨率范围筛选",
            example = "1920"
    )
    private Integer minWidth;

    @Schema(
            description = "最大图片宽度（像素），用于分辨率范围筛选",
            example = "3840"
    )
    private Integer maxWidth;

    @Schema(
            description = "最小图片高度（像素），用于分辨率范围筛选",
            example = "1080"
    )
    private Integer minHeight;

    @Schema(
            description = "最大图片高度（像素），用于分辨率范围筛选",
            example = "2160"
    )
    private Integer maxHeight;

    @Schema(
            description = "最小文件大小（字节），用于文件大小范围筛选",
            example = "1048576"
    )
    private Long minSize;

    @Schema(
            description = "最大文件大小（字节），用于文件大小范围筛选",
            example = "10485760"
    )
    private Long maxSize;

    @Schema(
            description = "排序字段：relevance-相关度，createTime-创建时间，usageCount-使用次数",
            example = "createTime",
            allowableValues = {"relevance", "createTime", "usageCount"}
    )
    private String sortBy;

    @Schema(
            description = "排序方向：asc-升序，desc-降序",
            example = "desc",
            allowableValues = {"asc", "desc"}
    )
    private String sortOrder;
}
