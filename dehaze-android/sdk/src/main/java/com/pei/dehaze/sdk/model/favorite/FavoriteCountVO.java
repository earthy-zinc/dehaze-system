package com.pei.dehaze.sdk.model.favorite;

import lombok.Data;

/**
 * 收藏数量统计（按类型分组）
 * 对齐后端 FavoriteCountVO（/api/v1/favorites/count）
 */
@Data
public class FavoriteCountVO {
    /** 收藏对象类型 */
    private String targetType;
    /** 该类型收藏数量 */
    private Long count;
}
