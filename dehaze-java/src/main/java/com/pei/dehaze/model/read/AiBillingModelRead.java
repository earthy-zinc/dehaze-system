package com.pei.dehaze.model.read;

import lombok.Data;

/** 用户模型消耗分布聚合行 */
@Data
public class AiBillingModelRead {

    private String model;

    private Long credits;

    private Long inputTokens;

    private Long outputTokens;
}
