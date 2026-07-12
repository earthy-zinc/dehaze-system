package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 智能推荐请求
 * 对齐后端 RecommendRequest
 */
@Data
public class RecommendRequest {
    /** 待去雾图片URL */
    private String imageUrl;
    /** 推荐数量（1-10） */
    private int topN = 3;
}
