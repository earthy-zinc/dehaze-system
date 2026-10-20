package com.pei.clarity.model.vo;

import lombok.Data;

/**
 * 样本级差异行
 *
 * @author clarity
 */
@Data
public class AiEvalSampleDiffItemVO {

    private Long sampleId;

    private String taskGoal;

    private Boolean currentPassed;

    private Boolean basePassed;

    private Double currentScore;

    private Double baseScore;

    private Double scoreDelta;
}
