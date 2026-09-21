package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 评测趋势行
 *
 * @author dehaze
 */
@Data
public class AiEvalTrendVO {

    private Long runId;

    private Long agentId;

    private String agentName;

    private String triggerType;

    private Integer status;

    private Double totalScore;

    private Map<String, Object> dimensions;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
