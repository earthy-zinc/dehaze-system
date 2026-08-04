package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

/**
 * 图像特征分析结果（7维特征向量）
 * 对齐后端 ImageFeatureAnalysisVO（/api/v1/recommendations/analyze）
 */
@Data
public class ImageFeatureAnalysisVO {
    /** 图像MD5值，用于关联推荐查询 */
    private String imageMd5;
    /** 雾霾浓度(light/moderate/heavy) */
    private String hazeLevel;
    /** 雾霾浓度置信度(0-1) */
    private Double hazeConfidence;
    /** 场景类型(urban/landscape/building/night/backlight/indoor) */
    private String sceneType;
    /** 场景置信度(0-1) */
    private Double sceneConfidence;
    /** 光照条件(bright/normal/dark/veryDark/backlight) */
    private String lighting;
    /** 图像复杂度(0-1) */
    private Double complexity;
    /** 颜色分布 */
    private ColorDistribution colorDistribution;
    /** 分辨率(sd/hd/uhd) */
    private String resolution;
    /** 噪声水平(low/medium/high) */
    private String noiseLevel;

    @Data
    public static class ColorDistribution {
        /** 色温 */
        private Double temperature;
        /** 饱和度 */
        private Double saturation;
    }
}
