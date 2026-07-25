package com.pei.dehaze.sdk.model.evaluation;

import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import lombok.Data;

import java.util.Map;

/**
 * 评估响应
 * <p>
 * 异步任务模式：POST 立即返回 logId + status=processing；
 * GET 轮询至 completed/failed 时返回完整字段。
 */
@Data
public class EvalResult {
    private Long logId;
    private PredEvalTaskStatus status;
    private Map<String, Double> metrics;
    private Integer time;
    private String errorMessage;
}
