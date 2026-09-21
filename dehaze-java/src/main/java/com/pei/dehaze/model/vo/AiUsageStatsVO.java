package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;

/**
 * AI 运营统计响应
 *
 * @author dehaze
 */
@Data
public class AiUsageStatsVO {

    private List<AiProviderHealthVO> providerHealth;

    private List<AiModelUsageVO> modelUsage;

    private AiDegradeFaultVO degradeFault;
}
