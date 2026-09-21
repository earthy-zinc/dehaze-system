package com.pei.dehaze.model.read;

import lombok.Data;

/** 过程链消耗聚合行（dimension 按请求维度承载 model/agent_code/user_id） */
@Data
public class AiTraceCostRead {

    private String dimension;

    private Long traceCount;

    private Long totalTokens;

    private Long promptTokens;

    private Long completionTokens;

    private Long cachedTokens;
}
