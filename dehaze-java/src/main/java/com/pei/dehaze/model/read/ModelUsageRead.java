package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * 模型用量聚合行
 *
 * @author dehaze
 */
@Data
public class ModelUsageRead {

    private String modelId;

    private Long callCount;

    private Long inputTokens;

    private Long outputTokens;

    private Long credits;
}
