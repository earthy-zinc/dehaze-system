package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/** 计费统计（按维度聚合） */
@Schema(description = "计费统计")
@Data
public class AiBillingStatVO {

    private String dimension;

    private Integer totalCredits;

    private Integer totalInputTokens;

    private Integer totalOutputTokens;

    /** 缓存命中率（chat 输入口径） */
    private Double cacheHitRate;

    private Integer creditsSaved;

    private Integer degradationCount;
}
