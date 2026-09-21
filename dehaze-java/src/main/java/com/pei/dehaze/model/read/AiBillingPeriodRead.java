package com.pei.dehaze.model.read;

import lombok.Data;

/** 用户消耗趋势聚合行（按日/月） */
@Data
public class AiBillingPeriodRead {

    private String date;

    private Long credits;

    private Long inputTokens;

    private Long outputTokens;

    private Long creditsSaved;

    private Long cachedInputTokens;
}
