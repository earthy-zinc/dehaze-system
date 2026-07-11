package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 预测日志查询参数
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "预测日志查询参数")
@Data
public class PredLogQuery {

    @Schema(description = "页码", defaultValue = "1")
    private Integer pageNum = 1;

    @Schema(description = "每页条数", defaultValue = "10")
    private Integer pageSize = 10;

    @Schema(description = "算法ID")
    private Long algorithmId;
}
