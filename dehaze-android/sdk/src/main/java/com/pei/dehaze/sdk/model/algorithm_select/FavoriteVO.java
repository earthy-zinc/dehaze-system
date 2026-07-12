package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 收藏VO
 * 对齐后端 FavoriteVO
 */
@Data
public class FavoriteVO {
    /** 收藏ID */
    private long id;
    /** 用户ID */
    private Long userId;
    /** 算法ID */
    private long algorithmId;
    /** 算法名称 */
    private String algorithmName;
    /** 收藏时间 */
    private String createTime;
}
