package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/** 性能趋势查询参数 */
@Schema(description = "性能趋势查询参数")
@Data
public class AiObservabilityTrendsQuery {

    /** model/agent */
    private String dimension = "model";

    private String startTime;

    private String endTime;
}
