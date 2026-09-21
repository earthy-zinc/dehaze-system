package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/**
 * 模型售价版本查询参数：沿用 python ModelPriceQuery 的 page/size 命名（非 pageNum/pageSize）。
 */
@Schema(description = "模型售价版本查询参数")
@Data
public class ModelPriceQuery {

    private String modelId;

    private Long providerId;

    @Min(value = 1, message = "页码必须大于0")
    private Integer page = 1;

    @Min(value = 1, message = "每页大小必须大于0")
    @Max(value = 100, message = "每页大小不能超过100")
    private Integer size = 20;
}
