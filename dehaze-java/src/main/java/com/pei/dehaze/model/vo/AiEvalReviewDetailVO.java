package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * 复核详情响应（样本定义 + 实际输出 + 四维得分说明）
 *
 * @author dehaze
 */
@Data
public class AiEvalReviewDetailVO {

    private Long runId;

    private Long agentId;

    private String agentName;

    private Long sampleId;

    private String taskGoal;

    private String allowedInput;

    private String expectedResult;

    private String expectedProcess;

    private String forbiddenBehavior;

    private List<String> tools;

    private String riskLevel;

    private Boolean judgePassed;

    private String actualOutput;

    private String error;

    private Map<String, Object> scores;

    private Map<String, Object> notes;
}
