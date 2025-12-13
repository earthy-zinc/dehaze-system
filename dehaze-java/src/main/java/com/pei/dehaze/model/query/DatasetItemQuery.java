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
@Schema(description = "图片搜索查询参数")
public class DatasetItemQuery extends BasePageQuery {

    @Schema(description = "数据集ID")
    private Long datasetId;

    @Schema(description = "关键词（文件名、描述、场景类型）")
    private String keywords;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "雾霾程度")
    private String hazeLevel;

    @Schema(description = "最小宽度")
    private Integer minWidth;

    @Schema(description = "最大宽度")
    private Integer maxWidth;

    @Schema(description = "最小高度")
    private Integer minHeight;

    @Schema(description = "最大高度")
    private Integer maxHeight;

    @Schema(description = "最小文件大小(字节)")
    private Long minSize;

    @Schema(description = "最大文件大小(字节)")
    private Long maxSize;

    @Schema(description = "排序字段（relevance/createTime/usageCount）")
    private String sortBy;

    @Schema(description = "排序方向（asc/desc）")
    private String sortOrder;
}
