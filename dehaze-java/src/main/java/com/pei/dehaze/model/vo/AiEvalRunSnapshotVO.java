package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 评测得分快照（对比用）
 *
 * @author dehaze
 */
@Data
public class AiEvalRunSnapshotVO {

    private Long runId;

    private Double totalScore;

    private Map<String, Object> dimensions;

    private Integer sampleCount;

    private Double passRate;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
