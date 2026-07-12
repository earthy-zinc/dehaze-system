package com.pei.dehaze.sdk.model.algorithm;

import lombok.Data;

/**
 * 算法收藏模型类
 */
@Data
public class AlgorithmFavorite {
    private Long id;
    private Long userId;
    private Long algorithmId;
    private String createTime;
}
