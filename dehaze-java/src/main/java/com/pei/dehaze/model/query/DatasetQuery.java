package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 数据集查询参数
 *
 * @author earthy-zinc
 * @since 2024-06-08 18:54:55
 */
@Schema(description = "数据集查询参数")
@Data
public class DatasetQuery {

    @Schema(
            description = "搜索关键字，支持按数据集名称、描述等字段模糊搜索",
            example = "测试数据集"
    )
    private String keyword;

    @Schema(
            description = "数据集类型筛选",
            example = "training"
    )
    private String type;

    @Schema(
            description = "状态筛选：1-启用，0-禁用",
            example = "1"
    )
    private Integer status;

    @Schema(
            description = "页码，默认1",
            example = "1"
    )
    private Integer pageNum = 1;

    @Schema(
            description = "每页大小，默认20",
            example = "20"
    )
    private Integer pageSize = 20;

}
