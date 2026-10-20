package com.pei.clarity.sdk.model.recommendation;

import lombok.Data;

/**
 * 推荐图像模型项
 * 对齐后端 RecommendedModelVO（/api/v1/recommendations/models）
 */
@Data
public class RecommendedModelVO {
    /** 推荐记录ID（sys_recommendation.id），用于提交反馈 */
    private Long recommendationId;
    /** 图像模型标识（统一模型规范 slug） */
    private String modelId;
    /** 模型名称 */
    private String modelName;
    /** 匹配度(0-100) */
    private Integer matchScore;
    /** 推荐理由 */
    private String reason;
    /** 模型评分(0-5) */
    private Double rating;
    /** 预估处理耗时(ms) */
    private Integer estimatedTime;
    /** 预期效果描述 */
    private String effectDescription;
}
