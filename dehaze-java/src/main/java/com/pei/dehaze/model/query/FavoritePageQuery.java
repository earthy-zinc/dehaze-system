package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "收藏分页查询参数")
public class FavoritePageQuery extends BasePageQuery {

    /** python favorite.py:26 默认 20（父类 BasePageQuery 为 10） */
    { setPageSize(20); }

    @Schema(description = "收藏对象类型筛选")
    private String targetType;

    @Schema(description = "关键词搜索（按收藏对象名称）")
    private String keywords;

    @Schema(description = "排序字段(createTime/rating/usageCount)")
    private String sortBy;

    @Schema(description = "排序方向(asc/desc)")
    private String sortOrder;
}
