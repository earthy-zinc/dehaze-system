package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 供应商健康看板行
 *
 * @author dehaze
 */
@Data
public class AiProviderHealthVO {

    private Long providerId;

    private String providerName;

    private String health;

    private Integer callCount;

    private Double successRate;

    private Double rate429;

    private Integer p95LatencyMs;

    private Boolean circuitOpen;
}
