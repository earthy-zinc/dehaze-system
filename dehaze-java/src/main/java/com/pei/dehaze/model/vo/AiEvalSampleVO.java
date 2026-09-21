package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 评测样本响应
 *
 * @author dehaze
 */
@Data
public class AiEvalSampleVO {

    private Long id;

    private Long datasetId;

    private String taskGoal;

    private String allowedInput;

    private List<String> tools;

    private String expectedProcess;

    private String expectedResult;

    private String forbiddenBehavior;

    private String riskLevel;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
