package com.pei.dehaze.model.vo;

import lombok.Data;

/**
 * 样本级差异行
 *
 * @author dehaze
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
