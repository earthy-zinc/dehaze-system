package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 评测执行记录响应
 *
 * @author dehaze
 */
@Data
public class AiEvalRunVO {

    private Long id;

    private Long agentId;

    private Long datasetId;

    private String triggerType;

    private Integer status;

    private Map<String, Object> scoreSummary;

    private List<Map<String, Object>> results;

    private Long createBy;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
