package com.pei.clarity.sdk.model.evaluation;

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
    /** 评估日志ID */
    private Long logId;
    /** 任务状态（1=处理中，2=已完成，3=失败） */
    private EvalTaskStatus status;
    /** 评估指标（completed 时返回） */
    private Map<String, Double> metrics;
    /** 预测图内容MD5（completed 时返回） */
    private String predMd5;
    /** 参考图内容MD5（completed 时返回） */
    private String gtMd5;
    /** 处理时间（秒） */
    private Integer time;
    /** 失败错误信息（failed 时返回） */
    private String errorMessage;
}
