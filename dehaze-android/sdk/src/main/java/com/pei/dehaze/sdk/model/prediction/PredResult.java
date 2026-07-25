package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

/**
 * 预测响应
 * <p>
 * 异步任务模式：POST 立即返回 logId + status=processing；
 * GET 轮询至 completed/failed 时返回完整字段。
 */
@Data
public class PredResult {
    private Long logId;
    private PredEvalTaskStatus status;
    private String resultUrl;
    private String resultThumbnailUrl;
    private Integer time;
    private String errorMessage;
}
