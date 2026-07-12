package com.pei.dehaze.sdk.model.evaluation;

import lombok.Data;

/**
 * 评估请求参数
 */
@Data
public class EvalParam {
    private Long algorithmId;
    private String predUrl;
    private String gtUrl;
}
