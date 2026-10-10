package com.pei.dehaze.model.vo;

import java.util.Map;
import lombok.Data;

/**
 * 两次评测对比响应
 *
 * @author dehaze
 */
@Data
public class AiEvalCompareVO {

    private Long runId;

    private Long baseRunId;

    private Long agentId;

    private AiEvalRunSnapshotVO current;

    private AiEvalRunSnapshotVO base;

    private Map<String, Object> dimensionDiff;

    private AiEvalSampleDiffVO sampleDiff;
}
