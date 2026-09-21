package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 资源消耗聚合查询参数 */
@Schema(description = "资源消耗聚合查询参数")
@Data
public class AiObservabilityCostsQuery {

    /** model/agent/user */
    private String dimension = "model";

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 10;

    private String startTime;

    private String endTime;
}
