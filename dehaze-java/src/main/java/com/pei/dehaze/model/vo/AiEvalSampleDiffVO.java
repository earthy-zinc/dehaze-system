package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;

/**
 * 样本级差异集合
 *
 * @author dehaze
 */
@Data
public class AiEvalSampleDiffVO {

    private List<AiEvalSampleDiffItemVO> added;

    private List<AiEvalSampleDiffItemVO> removed;

    private List<AiEvalSampleDiffItemVO> changed;

    private Integer unchangedCount;
}
