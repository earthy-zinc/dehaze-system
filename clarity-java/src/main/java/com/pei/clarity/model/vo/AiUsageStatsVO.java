package com.pei.clarity.model.vo;

import java.util.List;
import lombok.Data;

/**
 * AI 运营统计响应
 *
 * @author clarity
 */
@Data
public class AiUsageStatsVO {

    private List<AiProviderHealthVO> providerHealth;

    private List<AiModelUsageVO> modelUsage;

    private AiDegradeFaultVO degradeFault;
}
