package com.pei.dehaze.model.read;

import lombok.Data;

/** 过程链 Token 消耗按日趋势行 */
@Data
public class AiTraceCostTrendRead {

    private String date;

    private Long traceCount;

    private Long totalTokens;

    private Long promptTokens;

    private Long completionTokens;

    private Long cachedTokens;
}
