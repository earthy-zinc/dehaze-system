package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 收藏切换响应
 * 对齐后端 POST /algorithm-select/favorite 响应
 */
@Data
public class FavoriteToggleResult {
    /** 是否已收藏 */
    private boolean favorited;
    /** 收藏ID（收藏成功时返回） */
    private Long favoriteId;
}
