package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 余额流水查询参数 */
@Schema(description = "余额流水查询参数")
@Data
public class AiCreditLogQuery {

    @Min(1)
    private Long userId;

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 20;

    private String source;

    private String dateStart;

    private String dateEnd;
}
