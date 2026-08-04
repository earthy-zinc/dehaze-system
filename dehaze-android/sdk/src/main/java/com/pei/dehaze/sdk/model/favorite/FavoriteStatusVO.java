package com.pei.dehaze.sdk.model.favorite;

import lombok.Data;

/**
 * 收藏状态
 * 对齐后端 FavoriteStatusVO（/api/v1/favorites/{targetId}/status）
 */
@Data
public class FavoriteStatusVO {
    /** 收藏对象类型 */
    private String targetType;
    /** 收藏对象ID */
    private Long targetId;
    /** 是否已收藏 */
    private Boolean favorited;
}
