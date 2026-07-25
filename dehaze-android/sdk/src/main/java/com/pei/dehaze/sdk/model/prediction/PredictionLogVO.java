package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

/**
 * 预测日志VO（查询预测任务状态/日志列表）
 */
@Data
public class PredictionLogVO {
    private Long id;
    private Long algorithmId;
    private String originMd5;
    private String originUrl;
    private String predMd5;
    private String predUrl;
    private Integer time;
    private String status;
    private String errorMessage;
    private String createTime;
}
