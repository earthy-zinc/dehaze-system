package com.pei.dehaze.model.read;

import lombok.Data;

/** 计费统计聚合行（按 user/model/billType/day 维度） */
@Data
public class AiBillingStatRead {

    private String dimension;

    private Long totalCredits;

    private Long totalInputTokens;

    private Long totalOutputTokens;

    /** 仅 chat 类记录的缓存命中 token（命中率分母为 chat 输入） */
    private Long chatCachedTokens;

    private Long creditsSaved;

    private Long degradationCount;
}
