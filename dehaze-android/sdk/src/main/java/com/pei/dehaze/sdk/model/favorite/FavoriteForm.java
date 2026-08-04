package com.pei.dehaze.sdk.model.favorite;

import lombok.Data;

/**
 * 添加收藏表单
 * 对齐后端 FavoriteForm（/api/v1/favorites）
 */
@Data
public class FavoriteForm {
    /** 收藏对象类型(algorithm/result/dataset/image/preset) */
    private String targetType;
    /** 收藏对象ID */
    private Long targetId;
}
