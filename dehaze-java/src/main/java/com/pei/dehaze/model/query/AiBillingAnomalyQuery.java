package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 异常计费记录查询参数 */
@Schema(description = "异常计费记录查询参数")
@Data
public class AiBillingAnomalyQuery {

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 20;

    private Long userId;

    private String anomalyType;

    private Integer status;

    private String dateStart;

    private String dateEnd;
}
