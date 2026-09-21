package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 模型用量分布行
 *
 * @author dehaze
 */
@Data
public class AiModelUsageVO {

    private String modelId;

    private String displayName;

    private Long callCount;

    private Long inputTokens;

    private Long outputTokens;

    private Long credits;
}
