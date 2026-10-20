package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 模型用量聚合行
 *
 * @author clarity
 */
@Data
public class ModelUsageRead {

    private String modelId;

    private Long callCount;

    private Long inputTokens;

    private Long outputTokens;

    private Long credits;
}
