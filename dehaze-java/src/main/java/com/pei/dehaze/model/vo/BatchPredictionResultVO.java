package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Schema(description = "批量预测结果视图对象")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BatchPredictionResultVO {

    @Schema(description = "总数量")
    private int total;

    @Schema(description = "各图片预测结果")
    private List<PredictionResultVO> results;
}
