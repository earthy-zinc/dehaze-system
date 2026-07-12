package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

/**
 * 预测响应
 */
@Data
public class PredResult {
    private Long logId;
    private String resultUrl;
    private String resultThumbnailUrl;
    private Integer time;
    private Boolean fromCache;
}
