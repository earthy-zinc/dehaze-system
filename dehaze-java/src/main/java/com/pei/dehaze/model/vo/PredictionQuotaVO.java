package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Schema(description = "预测配额视图对象")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionQuotaVO {

    @Schema(description = "剩余次数")
    private int remaining;

    @Schema(description = "总次数")
    private int total;

    @Schema(description = "已使用次数")
    private int used;

    @Schema(description = "重置日期（下月1日）")
    private String resetDate;
}
