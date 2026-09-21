package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 评测总览行
 *
 * @author dehaze
 */
@Data
public class AiEvalOverviewVO {

    private Long agentId;

    private String agentCode;

    private String agentName;

    private Long runId;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime runTime;

    private String triggerType;

    private String gateStatus;

    private Double totalScore;

    private Map<String, Object> dimensions;

    private Boolean degraded;

    private Boolean highRiskFailed;
}
