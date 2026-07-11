package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "数据集查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class DatasetQuery extends BasePageQuery {

    @Schema(
            description = "搜索关键字，支持按数据集名称模糊搜索",
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

}
