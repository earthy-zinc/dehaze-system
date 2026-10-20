package com.pei.clarity.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/**
 * 图像任务历史查询参数
 */
@Schema(description = "图像任务历史查询参数")
@Data
public class PredLogQuery {

    @Schema(description = "页码", defaultValue = "1")
    @Min(value = 1, message = "页码必须大于0")
    private Integer pageNum = 1;

    @Schema(description = "每页条数", defaultValue = "10")
    @Min(value = 1, message = "每页大小必须大于0")
    @Max(value = 100, message = "每页大小不能超过100")
    private Integer pageSize = 10;

    @Schema(description = "图像模型标识筛选")
    private String model;
}
