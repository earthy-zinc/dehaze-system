package com.pei.clarity.model.vo;

import lombok.Data;

/**
 * 判分模型状态响应
 *
 * @author clarity
 */
@Data
public class AiJudgeStatusVO {

    private String consistencyState;

    private Boolean driftPaused;

    private Integer consistencyThreshold;

    private AiJudgeReviewStatsVO reviewStats;
}
