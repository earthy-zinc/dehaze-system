package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "图像特征分析结果（7维特征向量）")
public class ImageFeatureAnalysisVO {

    @Schema(description = "图像MD5值（基于imageUrl计算），用于关联推荐查询")
    private String imageMd5;

    @Schema(description = "雾霾浓度(light/moderate/heavy)")
    private String hazeLevel;

    @Schema(description = "雾霾浓度置信度(0-1)")
    private Double hazeConfidence;

    @Schema(description = "场景类型(urban/landscape/building/night/backlight/indoor)")
    private String sceneType;

    @Schema(description = "场景置信度(0-1)")
    private Double sceneConfidence;

    @Schema(description = "光照条件(bright/normal/dark/veryDark/backlight)")
    private String lighting;

    @Schema(description = "图像复杂度(0-1)")
    private Double complexity;

    @Schema(description = "颜色分布")
    private ColorDistribution colorDistribution;

    @Schema(description = "分辨率(sd/hd/uhd)")
    private String resolution;

    @Schema(description = "噪声水平(low/medium/high)")
    private String noiseLevel;

    @Data
    @Schema(description = "颜色分布")
    public static class ColorDistribution {

        @Schema(description = "色温")
        private Double temperature;

        @Schema(description = "饱和度")
        private Double saturation;
    }
}
