package com.pei.dehaze.sdk.model.favorite;

import lombok.Data;

/**
 * 收藏记录视图对象
 * 对齐后端 FavoriteVO（/api/v1/favorites/page）
 */
@Data
public class FavoriteVO {
    /** 收藏记录ID */
    private Long id;
    /** 用户ID */
    private Long userId;
    /** 收藏对象类型 */
    private String targetType;
    /** 收藏对象ID */
    private Long targetId;
    /** 收藏对象名称（关联查询） */
    private String targetName;
    /** 对象摘要 */
    private String targetSummary;
    /** 缩略图URL */
    private String targetThumbnail;
    /** 是否已失效（对象被删除） */
    private Boolean isInvalid;
    /** 收藏时间 */
    private String createTime;
}
