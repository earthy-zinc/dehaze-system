package com.pei.dehaze.sdk.model.favorite;

import com.pei.dehaze.sdk.model.PageQuery;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 收藏分页查询参数
 * 对齐后端 FavoritePageQuery（/api/v1/favorites/page）
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class FavoriteQuery extends PageQuery {
    /** 收藏对象类型筛选 */
    private String targetType;
    /** 关键词搜索（按收藏对象名称） */
    private String keywords;
    /** 排序字段(createTime/rating/usageCount) */
    private String sortBy;
    /** 排序方向(asc/desc) */
    private String sortOrder;
}
