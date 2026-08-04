package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

/**
 * 推荐算法项
 * 对齐后端 RecommendedAlgorithmVO（/api/v1/recommendations/algorithms）
 */
@Data
public class RecommendedAlgorithmVO {
    /** 推荐记录ID（sys_recommendation.id），用于提交反馈 */
    private Long recommendationId;
    /** 算法ID */
    private Long algorithmId;
    /** 算法名称 */
    private String algorithmName;
    /** 匹配度(0-100) */
    private Integer matchScore;
    /** 推荐理由 */
    private String reason;
    /** 算法评分(0-5) */
    private Double rating;
    /** 预估处理耗时(ms) */
    private Integer estimatedTime;
    /** 预期效果描述 */
    private String effectDescription;
}
