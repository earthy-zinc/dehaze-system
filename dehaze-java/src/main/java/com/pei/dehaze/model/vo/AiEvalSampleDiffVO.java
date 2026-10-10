package com.pei.dehaze.model.vo;

import java.util.List;
import lombok.Data;

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
