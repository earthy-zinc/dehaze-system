package com.pei.dehaze.sdk.model.evaluation;

import lombok.Data;

/**
 * 评估日志VO（查询评估任务状态/日志列表）
 */
@Data
public class EvaluationLogVO {
    private Long id;
    private Long algorithmId;
    private String predMd5;
    private String predUrl;
    private String gtMd5;
    private String gtUrl;
    private Integer time;
    private EvalResult result;
    private String createTime;
}
