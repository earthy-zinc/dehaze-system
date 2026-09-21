package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 成本单价列表查询参数 */
@Schema(description = "成本单价列表查询参数")
@Data
public class AiModelCostQuery {

    private String keyword;

    private String modelId;

    private Long providerId;

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 20;
}
