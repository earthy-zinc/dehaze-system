package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 复核队列项
 *
 * @author dehaze
 */
@Data
public class AiEvalReviewItemVO {

    private Long id;

    private Long runId;

    private Long sampleId;

    private Long agentId;

    private String agentName;

    private Boolean judgePassed;

    private String riskLevel;

    private Integer status;

    private Boolean agree;

    private String remark;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
