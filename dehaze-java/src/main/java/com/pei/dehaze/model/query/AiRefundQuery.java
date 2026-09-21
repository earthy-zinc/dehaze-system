package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 退款申请列表查询参数（管理端审核中心） */
@Schema(description = "退款申请列表查询参数")
@Data
public class AiRefundQuery {

    @Min(1)
    private Long userId;

    @Min(1)
    @Max(3)
    private Integer status;

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 20;

    private String dateStart;

    private String dateEnd;
}
