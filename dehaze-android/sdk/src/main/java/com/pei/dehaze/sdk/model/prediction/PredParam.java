package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

/**
 * 预测请求参数
 */
@Data
public class PredParam {
    private Long algorithmId;
    private String imageUrl;
    private String params;
}
