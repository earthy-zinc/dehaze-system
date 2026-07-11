package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 算法监控数据视图对象
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "算法监控数据视图对象")
@Data
public class AlgorithmMonitorVO {

    @Schema(description = "调用次数")
    private Long callCount;

    @Schema(description = "平均处理时间（毫秒）")
    private Double avgTime;

    @Schema(description = "成功率")
    private Double successRate;

    @Schema(description = "今日调用次数")
    private Long todayCallCount;
}
