package com.pei.dehaze.model.query;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
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
    @Min(value = 1, message = "页码必须大于0")
    private Integer pageNum = 1;

    @Schema(description = "每页条数", defaultValue = "10")
    @Min(value = 1, message = "每页大小必须大于0")
    @Max(value = 100, message = "每页大小不能超过100")
    private Integer pageSize = 10;

    @Schema(description = "算法ID")
    private Long algorithmId;
}
