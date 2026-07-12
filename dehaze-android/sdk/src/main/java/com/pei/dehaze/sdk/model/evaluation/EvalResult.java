package com.pei.dehaze.sdk.model.evaluation;

import lombok.Data;

import java.util.Map;

/**
 * 评估响应
 */
@Data
public class EvalResult {
    private Long logId;
    private Map<String, Double> metrics;
    private Boolean qualified;
    private Integer time;
}
