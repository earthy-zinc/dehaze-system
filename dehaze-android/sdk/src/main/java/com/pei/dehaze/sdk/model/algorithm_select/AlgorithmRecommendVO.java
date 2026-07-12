package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 算法推荐结果VO
 * 对齐后端 AlgorithmRecommendVO
 */
@Data
public class AlgorithmRecommendVO {
    /** 算法ID */
    private long algorithmId;
    /** 算法名称 */
    private String algorithmName;
    /** 匹配得分(0-100) */
    private double score;
    /** 推荐理由 */
    private String reason;
    /** 算法类型 */
    private String type;
}
